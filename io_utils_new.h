#ifndef HW2_IO_UTILS_H
#define HW2_IO_UTILS_H

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define X(i) (A_1 + (start_x_i + (i)) * h_1)
#define Y(i) (B_1 + (start_y_i + (i)) * h_2)

static int proc_rank = -1;
static int proc_column = -1;
static int proc_row = -1;
static int ranks_neigh[4];
static int border_count = -1;
static int cell_rows = -1;
static int cell_cols = -1;

static inline double get_cell(double *M, int row, int col)
{
	int idx = row * cell_cols + col;
	assert(idx <= cell_cols * cell_rows);
	return M[idx];
}

static inline double * get_row(double *M, int row)
{
	int idx = row * cell_cols;
	assert(idx + cell_cols <= cell_cols * cell_rows);
	return &M[idx];
}

static inline void set_cell(double *M, int row, int col, double value)
{
	int idx = row * cell_cols + col;
	assert(idx <= cell_cols * cell_rows);
	M[idx] = value;
}

static double *border_buffer_right = NULL;
static double *border_buffer_left = NULL;

static int border_size[4];

static int start_x_i = -1;
static int start_y_i = -1;

enum {
	BOTTOM_BORDER, TOP_BORDER, LEFT_BORDER, RIGHT_BORDER
};

static bool is_top = false;
static bool is_bottom = false;
static bool is_left = false;
static bool is_right = false;

static inline double global_scalar_fraction(double local_numerator, double local_denominator)
{
	double local_buf[2], global_buf[2];
	local_buf[0] = local_numerator;
	local_buf[1] = local_denominator;
	int rc = MPI_Allreduce(local_buf, global_buf, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	(void) rc;
	assert(rc == MPI_SUCCESS);
	return global_buf[0] / global_buf[1];
}

/**
 * Global increment: || P_i+1 - P_i ||.
 * local_increment of P_i+1 - P_i.
 * local_error of P (сравнение с точным решением).
 * global_error (все процессы): || exact - P ||.
 *
 * Global increment.
 */
static inline double global_increment(double local_increment, double local_error, double *global_error, double step)
{
	double local_buf[2];
	double global_buf[2];
	local_buf[0] = local_increment;
	local_buf[1] = local_error;
	int rc = MPI_Allreduce(local_buf, global_buf, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	(void) rc;
	assert(rc == MPI_SUCCESS);
	*global_error = sqrt(global_buf[1] * step);
	return sqrt(global_buf[0] * step);
}

/**
 * Передача сообщений (границ матрицы) без блокировки (асинхронная передача) соседнему процессу
 */
static inline void send_borders(double *matrix)
{
	for (int i = 0; i < 4; ++i) {
		MPI_Request req;
		if (ranks_neigh[i] == -1)
			continue;
		double *to_send;
		if (i == BOTTOM_BORDER || i == TOP_BORDER) {
			if (i == BOTTOM_BORDER)
				to_send = get_row(matrix, 0);
			else
				to_send = get_row(matrix, cell_rows - 1);
		} else {
			int col;
			if (i == LEFT_BORDER) {
				col = 0;
				to_send = border_buffer_left;
			} else {
				col = cell_cols - 1;
				to_send = border_buffer_right;
			}
			for (int i = 0; i < cell_rows; ++i)
				to_send[i] = get_cell(matrix, i, col);
		}
		int size = border_size[i];
		int rc = MPI_Isend(to_send, size, MPI_DOUBLE, ranks_neigh[i], i, MPI_COMM_WORLD, &req);
		(void) rc; 
		assert(rc == MPI_SUCCESS);
		MPI_Request_free(&req);
	}
}

/**
 * Прием сообщений (границ матрицы) без блокировки (асинхронная передача) от соседнего процесса
 */
static inline void receive_borders(double **borders, MPI_Request *reqs)
{
	for (int i = 0; i < 4; ++i) {
		if (ranks_neigh[i] == -1)
			continue;
		int type_from;
		if (i == BOTTOM_BORDER || i == TOP_BORDER) {
			if (i == BOTTOM_BORDER)
				type_from = TOP_BORDER;
			else
				type_from = BOTTOM_BORDER;
		} else {
			if (i == LEFT_BORDER)
				type_from = RIGHT_BORDER;
			else
				type_from = LEFT_BORDER;
		}
		int size = border_size[i];
		int rc = MPI_Irecv(borders[i], size, MPI_DOUBLE, ranks_neigh[i],
				   type_from, MPI_COMM_WORLD, &reqs[i]);
		(void) rc;
		assert(rc == MPI_SUCCESS);
	}
}

static inline void sync_receive_borders(MPI_Request *reqs, int count)
{
	int rc = MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);
	(void) rc;
	assert(rc == MPI_SUCCESS);
}

/**
 * Функция распределения матрицы по процессам 
 **/
static inline int calculate_cells(int table_height, int table_width, int proc_count)
{
	int point_rest_row = 0;               //точки, которые не попали в блоки при разбиении нацело на 2
	int point_rest_col = 0;
	int cell_width_cur = table_width;	   //ширина и высота блока при идеальном разбиении (обычно всё не так хорошо)
	int cell_height_cur = table_height;
	int n = 1;                             //степень для определения количества процессов, которые на данный момент получили блоки
	int m = 0;                             //степень для подсчета оставшихся без блока точек
	int l = 0;
	int proc_cur = 1;
	int proc_cur_col = 1;
	int proc_cur_row = 1;	
	int cell_old_width;
	int cell_old_heigh;
	int col_proc_count = 1;                //процессорная сетка
	int row_proc_count = 1;
	while (proc_cur < proc_count)
	{
		cell_old_width = cell_width_cur;
		cell_width_cur = cell_width_cur / 2;
		if (cell_old_width % 2 != 0)
			point_rest_col += (int)pow(2,(double)m);
		m++;
		proc_cur = (int)pow(2,(double)n);
		proc_cur_col = proc_cur_col * 2;
		n++;
		if (proc_cur < proc_count)
		{
			cell_old_heigh = cell_height_cur;
			cell_height_cur = cell_height_cur / 2;
			if (cell_old_heigh % 2 != 0)
				point_rest_row += (int)pow(2,(double)l);
			l++;
			proc_cur = (int)pow(2,(double)n);
			proc_cur_row = proc_cur_row * 2;
			n++;
		}
	}
	col_proc_count = proc_cur_col;
	row_proc_count = proc_cur_row;
        proc_row = proc_rank / col_proc_count;         //координаты процесса в процессорной сетке
	proc_column = proc_rank % col_proc_count;	   //координаты процесса в процессорной сетке	


	/* После определения общего размера блоков разбиения, смотрим, 
	 * остались ли у нас точки, не попавшие ни в один блок. То есть 
	 * мы не смогли разбить матрицу на блоки одинакового размера.
	 * По идее, мы просто расширяем длину и ширину верхних левых блоков
	 * на одну точку (если предвставлять общую картину - столбцу/строке).
	 */
	 
	/* Определяем длину/ ширину блока текущего процесса*/
	if (point_rest_col != 0)
	{
		if (proc_column < point_rest_col)
			cell_cols = cell_width_cur + 1;
		else 
			cell_cols = cell_width_cur;
	}
	else 
		cell_cols = cell_width_cur;
	if (point_rest_row != 0)
	{
		if (proc_row < point_rest_row)
			cell_rows = cell_height_cur + 1;
		else
			cell_rows = cell_height_cur;
	}
	else 
		cell_rows = cell_height_cur;
	 
	/* Самое интересное, теперь определяем координаты начальных элементов каждого процесса*/ 
	
	if (point_rest_col != 0)
	{
		if (proc_column <= point_rest_col)
			start_x_i = (cell_width_cur + 1)*proc_column;
		else 
			start_x_i = (cell_width_cur + 1)*point_rest_col + cell_width_cur * (proc_column - point_rest_col);
	}
	else 
		start_x_i = cell_width_cur * proc_column;
	
	if (point_rest_row != 0)
	{
		if (proc_row <= point_rest_row)
			start_y_i = (cell_height_cur + 1)*proc_row;
		else 
			start_y_i = (cell_height_cur + 1)*point_rest_row + cell_height_cur * (proc_row - point_rest_row);
	}
	else 
		start_y_i = cell_height_cur * proc_row;
	
	/*Определяем, принадлежит ли конкретный блок процесса границе*/
	
	border_count = 4;				  
	if (proc_row + 1 == row_proc_count) {
		is_top = true;
		--border_count;
	}
	if (proc_row == 0) {
		is_bottom = true;
		--border_count;
	}

	if (proc_column == 0) {
		is_left = true;
		--border_count;
	}
	if (proc_column + 1 == col_proc_count) {
		is_right = true;
		--border_count;
	}
	
	
	if (! is_top) {							                        //Заполняем данные по соседним процессам, если эти процессы 
		ranks_neigh[TOP_BORDER] = proc_rank + col_proc_count;		//существуют. Какой процесс сосед? Именнр порядковый номер
		border_size[TOP_BORDER] = cell_cols;
	} else {
		ranks_neigh[TOP_BORDER] = -1;
		border_size[TOP_BORDER] = -1;
	}
	if (! is_bottom) {
		ranks_neigh[BOTTOM_BORDER] = proc_rank - col_proc_count;
		border_size[BOTTOM_BORDER] = cell_cols;
	} else {
		ranks_neigh[BOTTOM_BORDER] = -1;
		border_size[BOTTOM_BORDER] = -1;
	}
	if (! is_left) {
		ranks_neigh[LEFT_BORDER] = proc_rank - 1;
		border_size[LEFT_BORDER] = cell_rows;
	} else {
		ranks_neigh[LEFT_BORDER] = -1;
		border_size[LEFT_BORDER] = -1;
	}
	if (! is_right) {
		ranks_neigh[RIGHT_BORDER] = proc_rank + 1;
		border_size[RIGHT_BORDER] = cell_rows;
	} else {
		ranks_neigh[RIGHT_BORDER] = -1;
		border_size[RIGHT_BORDER] = -1;
	}
	return 0;
}

#endif
