#include "io_utils_new.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>


/** Задаем границы прямоугольной области */
static const double A_1 = -1, A_2 = 2, B_1 = -2, B_2 = 2;
/** Инициализируем шаг по сетке */ 
static double h_1 = 0;
static double h_2 = 0;
/** Необходимые константы для упрощения вычислений coef_h_1 = 1/(h_1)^2; coef_h_2 = 1/(h_2)^2*/
static double coef_h_1 = 0;
static double coef_h_2 = 0;

/** Переменные для подсчета времени выполнения программы */
static double start_time = 0;
static double end_time = 0;

/** Наперед заданная точность */
static double epsilon = 0.00001;

static int first_internal_row = -1;
static int first_second_internal_row = -1;
static int last_internal_row = -1;
static int last_second_internal_row = -1;
static int first_internal_col = -1;
static int first_second_internal_col = -1;
static int last_internal_col = -1;
static int last_second_internal_col = -1;

/** Матрицы для хранения данных, использующиеся в расчетах алгоритма. */
static double *P_neigh[4];
static double *P = NULL;
static double *P_prev = NULL;
static double *R_neigh[4];
static double *R = NULL;

/** Буфер для хранения промежуточных результатов. */
static double *matrix_buffer = NULL;


static int iterations_count = 0;
static const int max_iterations_count = 2000000;    //на всякий случай, если что-то пойдет не так

static inline double
phi(double x, double y)
{
	return exp(1 - (x + y)*(x + y));
}

static inline double
exact(double x, double y)
{
	return phi(x, y);
}

static inline double
F(double x, double y)
{
	double temp = (x + y)*(x + y);
	return 4 * (-2*temp + 1) * exp(1-temp);
}

/** Выделяем память под границы. */
static inline void create_borders(int border_id, bool exists)
{
	if (exists) {
		int elem_count = border_size[border_id];
		int size = elem_count * 2;
		double *mem = (double *) calloc(size, sizeof(double));
		assert(mem != NULL);
		P_neigh[border_id] = mem;
		R_neigh[border_id] = mem + elem_count;
	} else {
		P_neigh[border_id] = NULL;
		R_neigh[border_id] = NULL;
	}
}

static inline double local_scalar(double *a, double *b)
{
	double ret = 0;
	#pragma omp parallel for reduction(+:ret)
	for (int i = first_internal_row; i <= last_internal_row; ++i) {
		for (int j = first_internal_col; j <= last_internal_col; ++j)
			ret += get_cell(a, i, j) * get_cell(b, i, j);
	}
	return ret * h_1 * h_2;
}


static inline double laplas_5(double *matrix, int row, int col, double **borders)
{
	double a_11 = 0, a_01 = 0, a_21 = 0, a_10 = 0, a_12 = 0; 
	double tmp1 = 0;
	double tmp2 = 0;
	assert(row >= first_internal_row && row <= last_internal_row);
	assert(col >= first_internal_col && col <= last_internal_col);
	a_11 = get_cell(matrix, row, col);
	/* для самых внутренних точек */
	if ((row >= first_second_internal_row) && (row <= last_second_internal_row) && (col >= first_second_internal_col) && (col <= last_second_internal_col))
	{
		if (row > 0)
			a_10 = get_cell(matrix, row - 1, col);
		else{
			assert(borders[BOTTOM_BORDER] != NULL);
			a_10 = borders[BOTTOM_BORDER][col];
		}
		if (row < cell_rows - 1) 
			a_12 = get_cell(matrix, row + 1, col);
		else {
			assert(borders[TOP_BORDER] != NULL);
			a_12 = borders[TOP_BORDER][col];
		}
		if (col > 0)
			a_01 = get_cell(matrix, row, col - 1);
		else {
			assert(borders[LEFT_BORDER] != NULL);
			a_01 = borders[LEFT_BORDER][row];
		}
		if (col < cell_cols - 1)
			a_21 = get_cell(matrix, row, col + 1);
		else {
			assert(borders[RIGHT_BORDER] != NULL);
			a_21 = borders[RIGHT_BORDER][row];
		}

		tmp1 = coef_h_1 * (a_01 - 2 * a_11 + a_21);
		tmp2 = coef_h_2 * (a_10 - 2 * a_11 + a_12);
	}
	/* левые внутренние точки */
	else if (col < first_second_internal_col)
	{
		/* угловая внутренняя точка на пересечении верхней и левой границы */
		if (row < first_second_internal_row)
		{
			if (col < cell_cols - 1)
				a_21 = get_cell(matrix, row, col + 1);
			else {
				assert(borders[RIGHT_BORDER] != NULL);
				a_21 = borders[RIGHT_BORDER][row];
			}
			if (row < cell_rows - 1) 
				a_12 = get_cell(matrix, row + 1, col);
			else {
				assert(borders[TOP_BORDER] != NULL);
				a_12 = borders[TOP_BORDER][col];
			}
			tmp1 = coef_h_1 * (-2 * a_11 + a_21);
			tmp2 = coef_h_2 * (-2 * a_11 + a_12);
		}
		/* угловая внутренняя точка на пересечении нижней и левой границы */
		else if (row > last_second_internal_row)
		{
			if (row > 0)
				a_10 = get_cell(matrix, row - 1, col);
			else{
				assert(borders[BOTTOM_BORDER] != NULL);
				a_10 = borders[BOTTOM_BORDER][col];
			}
			if (col < cell_cols - 1)
				a_21 = get_cell(matrix, row, col + 1);
			else {
				assert(borders[RIGHT_BORDER] != NULL);
				a_21 = borders[RIGHT_BORDER][row];
			}
			tmp1 = coef_h_1 * (-2 * a_11 + a_21);
			tmp2 = coef_h_2 * (-2 * a_11 + a_10);
		}
		else
		{
			/*все остальные левые точки*/
			if (col < cell_cols - 1)
				a_21 = get_cell(matrix, row, col + 1);
			else {
				assert(borders[RIGHT_BORDER] != NULL);
				a_21 = borders[RIGHT_BORDER][row];
			}
			if (row > 0)
				a_10 = get_cell(matrix, row - 1, col);
			else{
				assert(borders[BOTTOM_BORDER] != NULL);
				a_10 = borders[BOTTOM_BORDER][col];
			}
			if (row < cell_rows - 1) 
				a_12 = get_cell(matrix, row + 1, col);
			else {
				assert(borders[TOP_BORDER] != NULL);
				a_12 = borders[TOP_BORDER][col];
			}
			tmp1 = coef_h_1 * (-2 * a_11 + a_21);
			tmp2 = coef_h_2 * (a_10 - 2 * a_11 + a_12);
		}
	}
	/* правые внутренние точки */
	else if (col > last_second_internal_col)
	{
		/* угловая внутренняя точка на пересечении верхней и правой границы */
		if (row < first_second_internal_row)
		{
			if (col > 0)
				a_01 = get_cell(matrix, row, col - 1);
			else {
				assert(borders[LEFT_BORDER] != NULL);
				a_01 = borders[LEFT_BORDER][row];
			}
			if (row < cell_rows - 1) 
				a_12 = get_cell(matrix, row + 1, col);
			else {
				assert(borders[TOP_BORDER] != NULL);
				a_12 = borders[TOP_BORDER][col];
			}
			tmp1 = coef_h_1 * (-2 * a_11 + a_01);
			tmp2 = coef_h_2 * (-2 * a_11 + a_12);		
		}
		/* угловая внутренняя точка на пересечении нижней и правой границы */
		else if (row > last_second_internal_row)
		{
			if (row > 0)
				a_10 = get_cell(matrix, row - 1, col);
			else{
				assert(borders[BOTTOM_BORDER] != NULL);
				a_10 = borders[BOTTOM_BORDER][col];
			}
			if (col > 0)
				a_01 = get_cell(matrix, row, col - 1);
			else {
				assert(borders[LEFT_BORDER] != NULL);
				a_01 = borders[LEFT_BORDER][row];
			}
			tmp1 = coef_h_1 * (-2 * a_11 + a_01);
			tmp2 = coef_h_2 * (-2 * a_11 + a_10);
		}
		else
		{
			if (row > 0)
				a_10 = get_cell(matrix, row - 1, col);
			else{
				assert(borders[BOTTOM_BORDER] != NULL);
				a_10 = borders[BOTTOM_BORDER][col];
			}
			if (col > 0)
				a_01 = get_cell(matrix, row, col - 1);
			else {
				assert(borders[LEFT_BORDER] != NULL);
				a_01 = borders[LEFT_BORDER][row];
			}
			if (row < cell_rows - 1) 
				a_12 = get_cell(matrix, row + 1, col);
			else {
				assert(borders[TOP_BORDER] != NULL);
				a_12 = borders[TOP_BORDER][col];
			}
			tmp1 = coef_h_1 * (a_01 - 2 * a_11);
			tmp2 = coef_h_2 * (a_10 - 2 * a_11 + a_12);
		}
	}
	/* внутренние верхние точки */
	else if (row < first_second_internal_row)
	{
		if (row < cell_rows - 1) 
			a_12 = get_cell(matrix, row + 1, col);
		else {
			assert(borders[TOP_BORDER] != NULL);
			a_12 = borders[TOP_BORDER][col];
		}
		if (col > 0)
			a_01 = get_cell(matrix, row, col - 1);
		else {
			assert(borders[LEFT_BORDER] != NULL);
			a_01 = borders[LEFT_BORDER][row];
		}
		if (col < cell_cols - 1)
			a_21 = get_cell(matrix, row, col + 1);
		else {
			assert(borders[RIGHT_BORDER] != NULL);
			a_21 = borders[RIGHT_BORDER][row];
		}
		tmp1 = coef_h_1 * (a_01 - 2 * a_11 + a_21);
		tmp2 = coef_h_2 * (-2 * a_11 + a_12);		
	}
	else if (row > last_second_internal_row)
	{
		if (col > 0)
			a_01 = get_cell(matrix, row, col - 1);
		else {
			assert(borders[LEFT_BORDER] != NULL);
			a_01 = borders[LEFT_BORDER][row];
		}
		if (col < cell_cols - 1)
			a_21 = get_cell(matrix, row, col + 1);
		else {
			assert(borders[RIGHT_BORDER] != NULL);
			a_21 = borders[RIGHT_BORDER][row];
		}
		if (row > 0)
			a_10 = get_cell(matrix, row - 1, col);
		else{
			assert(borders[BOTTOM_BORDER] != NULL);
			a_10 = borders[BOTTOM_BORDER][col];
		}
		tmp1 = coef_h_1 * (a_01 - 2 * a_11 + a_21);
		tmp2 = coef_h_2 * (-2 * a_11 + a_10);
	}
	return tmp1 + tmp2;
}

static inline void laplas_5_matrix(double *src, double **src_borders, double *dst)
{
	#pragma omp parallel for
	for (int i = first_internal_row; i <= last_internal_row; ++i) {
		for (int j = first_internal_col; j <= last_internal_col; ++j)
			set_cell(dst, i, j, -laplas_5(src, i, j, src_borders));
	}
}

/**
 * P_next = P - tau * R.
 * На выходе out_error локальная ошибка P.
 */
static inline double
calculate_next_P(double tau, double *discrepancy_matrix,
		 double **discrepancy_borders, double *out_error, double step)
{
	double error = 0;
	double increment = 0;
	#pragma omp parallel for reduction(+:increment, error)
	for (int i = 0; i < cell_rows; ++i) {
		for (int j = 0; j < cell_cols; ++j) {
			double old = get_cell(P, i, j);
			set_cell(P, i, j, old -
				 tau * get_cell(discrepancy_matrix, i, j));
			if (i >= first_internal_row && i <= last_internal_row &&
			    j >= first_internal_col && j <= last_internal_col) {
				double cell = get_cell(P, i, j);
				double local_increment = cell - old;
				increment += local_increment * local_increment;
				double local_error = cell - exact(X(j), Y(i));
				error += local_error * local_error;
			}
		}
	}
	#pragma omp parallel for
	for (int i = 0; i < 4; ++i) {
		if (P_neigh[i] == NULL)
			continue;
		assert(discrepancy_borders[i] != NULL);
		int count = border_size[i];
		for (int j = 0; j < count; ++j)
			P_neigh[i][j] -= tau * discrepancy_borders[i][j];
	}
	*out_error = error;
	return increment;
}

/**
 * R = -laplas_5(P) - F() для внутренних точек и
 * R = 0 для границы.
 */
static inline void
calculate_next_R()
{
	/* Сейчас будет жесть. Уточняю алгоритм для идеальной сходимости */

	/* Совсем внутренние точки */
	#pragma omp parallel for
	for (int i = first_second_internal_row; i <= last_second_internal_row; ++i) {
		for (int j = first_second_internal_col; j <= last_second_internal_col; ++j) {
			set_cell(R, i, j, -laplas_5(P, i, j, P_neigh) - F(X(j), Y(i)));
		}
	}

	if (is_bottom)
	{	
		/* Внутренние точки вблизи верхней границы */
		#pragma omp parallel for
		for (int j = first_second_internal_col; j <= last_second_internal_col; ++j) {
			set_cell(R, 1, j, -laplas_5(P, 1, j, P_neigh) - F(X(j), Y(1)) - phi(X(j), Y(0)) * coef_h_2);
		}
		if (is_left)
		{
			/* угловая на пересечении левой и верхней границы */
			set_cell(R, 1, 1, -laplas_5(P, 1, 1, P_neigh) - F(X(1), Y(1)) - phi(X(1), Y(0)) * coef_h_2 - phi(X(0), Y(1)) * coef_h_1);
		}
		if (is_right)
		{
			/* угловая на пересечении правой и верхней границы */
			set_cell(R, 1, cell_cols - 2, -laplas_5(P, 1, cell_cols - 2, P_neigh) - F(X(cell_cols - 2), Y(1)) - phi(X(cell_cols - 2), Y(0)) * coef_h_2 - phi(X(cell_cols - 1), Y(1)) * coef_h_1);
		}
	}

	if (is_top)
	{
		/* Внутренние точки вблизи нижней границы */
		#pragma omp parallel for
		for (int j = first_second_internal_col; j <= last_second_internal_col; ++j) {
			set_cell(R, cell_rows - 2, j, -laplas_5(P, cell_rows - 2, j, P_neigh) - F(X(j), Y(cell_rows - 2)) - phi(X(j), Y(cell_rows - 1)) * coef_h_2);
		}
		if (is_left)
		{
			/* угловая на пересечении левой и нижней границы */
			set_cell(R, cell_rows - 2, 1, -laplas_5(P, cell_rows - 2, 1, P_neigh) - F(X(1), Y(cell_rows - 2)) - phi(X(1), Y(cell_rows - 1)) * coef_h_2 - phi(X(0), Y(cell_rows - 2)) * coef_h_1);
		}
		if (is_right)
		{
			/* угловая на пересечении правой и верхней границы */
			set_cell(R, cell_rows - 2, cell_cols - 2, -laplas_5(P, cell_rows - 2, cell_cols - 2, P_neigh) - F(X(cell_cols - 2), Y(cell_rows - 2)) - phi(X(cell_cols - 2), Y(cell_rows - 1)) * coef_h_2 - phi(X(cell_cols - 1), Y(cell_rows - 2)) * coef_h_1);
		}
		
	}

	if (is_left)
	{
		/*Внутренние точки вблизи левой границы */
		#pragma omp parallel for
		for (int i = first_second_internal_row; i <= last_second_internal_row; ++i) {
			set_cell(R, i, 1, -laplas_5(P, i, 1, P_neigh) - F(X(1), Y(i)) - phi(X(0), Y(i)) * coef_h_1);
		}		
	}

	if (is_right)
	{
		/*Внутренние точки вблизи левой границы */
		#pragma omp parallel for
		for (int i = first_second_internal_row; i <= last_second_internal_row; ++i) {
			set_cell(R, i, cell_cols - 2, -laplas_5(P, i, cell_cols - 2, P_neigh) - F(X(cell_cols - 2), Y(i)) - phi(X(cell_cols - 1), Y(i)) * coef_h_1);
		}		
	}

	#pragma omp parallel if (border_count <= 2)
	#pragma omp sections
	{
		#pragma omp section
		{
			if (is_bottom) {
				for (int i = 0; i < cell_cols; ++i)
					set_cell(R, 0, i, 0);
			}
		}
		#pragma omp section
		{
			if (is_top) {
				for (int i = 0; i < cell_cols; ++i)
					set_cell(R, cell_rows - 1, i, 0);
			}
		}
		#pragma omp section
		{
			if (is_left) {
				for (int i = 0; i < cell_rows; ++i)
					set_cell(R, i, 0, 0);
			}
		}
		#pragma omp section
		{
			if (is_right) {
				for (int i = 0; i < cell_rows; ++i)
					set_cell(R, i, cell_cols - 1, 0);
			}
		}
	}
}

/**
 * Выделение памяти под матрицы.
 */
static inline void
create_matrices()
{
	int max_count = cell_rows > cell_cols ? cell_rows : cell_cols;
	int size = cell_rows * cell_cols;
	double *mem = (double *) calloc(size * 4 + max_count * 2, sizeof(double));
	assert(mem != NULL);
	P = mem;
	R = mem + size;
	matrix_buffer = mem + size * 2;
	P_prev = mem + size * 3;
	border_buffer_left = mem + size * 4;
	border_buffer_right = mem + size * 4 + max_count;
	create_borders(BOTTOM_BORDER, ! is_bottom);
	create_borders(TOP_BORDER, ! is_top);
	create_borders(LEFT_BORDER, ! is_left);
	create_borders(RIGHT_BORDER, ! is_right);

	/* Заполняем начальное приближение P граничными значениями */
	if (is_bottom) {
		for (int i = 0; i < cell_cols; ++i)
			set_cell(P, 0, i, phi(X(i), Y(0)));
	}
	if (is_top) {
		for (int i = 0; i < cell_cols; ++i) {
			set_cell(P, cell_rows - 1, i, phi(X(i), Y(cell_rows - 1)));
		}
	}
	if (is_right) {
		for (int i = 0; i < cell_rows; ++i) {
			set_cell(P, i, cell_cols - 1, phi(X(cell_cols - 1), Y(i)));
		}
	}
	if (is_left) {
		for (int i = 0; i < cell_rows; ++i)
			set_cell(P, i, 0, phi(X(0), Y(i)));
	}
	MPI_Request req[4];                    //Идентификатор асинхронной передачи
	for (int i = 0; i < 4; ++i)
		req[i] = MPI_REQUEST_NULL;
	send_borders(P);		       //Посылаем границы P соседним процессам
	receive_borders(P_neigh, req);	       //Получаем границы от соседних процессов
	sync_receive_borders(req, 4);	       //Ждем завершения, пока 4 запроса будут обработаны

	/* Внутренние индексы блока для каждого процесса по отдельности. */
	if (is_bottom){
		first_internal_row = 1;
		first_second_internal_row = 2;
	}
	else{
		first_internal_row = 0;
		first_second_internal_row = 0;
	}
	if (is_top){
		last_internal_row = cell_rows - 2;
		last_second_internal_row = cell_rows - 3;
	}
	else{
		last_internal_row = cell_rows - 1;
		last_second_internal_row = cell_rows - 1;
	}
	if (is_left){
		first_internal_col = 1;
		first_second_internal_col = 2;
	}
	else{
		first_internal_col = 0;
		first_second_internal_col = 0;
	}
	if (is_right){
		last_internal_col = cell_cols - 2;
		last_second_internal_col = cell_cols - 3;
	}
	else{
		last_internal_col = cell_cols - 1;
		last_second_internal_col = cell_cols - 1;
	} 

	/* Сейчас будет жесть. Уточняю алгоритм для идеальной сходимости */

	/* Совсем внутренние точки */
	#pragma omp parallel for
	for (int i = first_second_internal_row; i <= last_second_internal_row; ++i) {
		for (int j = first_second_internal_col; j <= last_second_internal_col; ++j) {
			set_cell(R, i, j, -laplas_5(P, i, j, P_neigh) - F(X(j), Y(i)));
		}
	}

	if (is_bottom)
	{	
		/* Внутренние точки вблизи верхней границы */
		#pragma omp parallel for
		for (int j = first_second_internal_col; j <= last_second_internal_col; ++j) {
			set_cell(R, 1, j, -laplas_5(P, 1, j, P_neigh) - F(X(j), Y(1)) - phi(X(j), Y(0)) * coef_h_2);
		}
		if (is_left)
		{
			/* угловая на пересечении левой и верхней границы */
			set_cell(R, 1, 1, -laplas_5(P, 1, 1, P_neigh) - F(X(1), Y(1)) - phi(X(1), Y(0)) * coef_h_2 - phi(X(0), Y(1)) * coef_h_1);
		}
		if (is_right)
		{
			/* угловая на пересечении правой и верхней границы */
			set_cell(R, 1, cell_cols - 2, -laplas_5(P, 1, cell_cols - 2, P_neigh) - F(X(cell_cols - 2), Y(1)) - phi(X(cell_cols - 2), Y(0)) * coef_h_2 - phi(X(cell_cols - 1), Y(1)) * coef_h_1);
		}
	}

	if (is_top)
	{
		/* Внутренние точки вблизи нижней границы */
		#pragma omp parallel for
		for (int j = first_second_internal_col; j <= last_second_internal_col; ++j) {
			set_cell(R, cell_rows - 2, j, -laplas_5(P, cell_rows - 2, j, P_neigh) - F(X(j), Y(cell_rows - 2)) - phi(X(j), Y(cell_rows - 1)) * coef_h_2);
		}
		if (is_left)
		{
			/* угловая на пересечении левой и нижней границы */
			set_cell(R, cell_rows - 2, 1, -laplas_5(P, cell_rows - 2, 1, P_neigh) - F(X(1), Y(cell_rows - 2)) - phi(X(1), Y(cell_rows - 1)) * coef_h_2 - phi(X(0), Y(cell_rows - 2)) * coef_h_1);
		}
		if (is_right)
		{
			/* угловая на пересечении правой и верхней границы */
			set_cell(R, cell_rows - 2, cell_cols - 2, -laplas_5(P, cell_rows - 2, cell_cols - 2, P_neigh) - F(X(cell_cols - 2), Y(cell_rows - 2)) - phi(X(cell_cols - 2), Y(cell_rows - 1)) * coef_h_2 - phi(X(cell_cols - 1), Y(cell_rows - 2)) * coef_h_1);
		}
		
	}

	if (is_left)
	{
		/*Внутренние точки вблизи левой границы */
		#pragma omp parallel for
		for (int i = first_second_internal_row; i <= last_second_internal_row; ++i) {
			set_cell(R, i, 1, -laplas_5(P, i, 1, P_neigh) - F(X(1), Y(i)) - phi(X(0), Y(i)) * coef_h_1);
		}		
	}

	if (is_right)
	{
		/*Внутренние точки вблизи левой границы */
		#pragma omp parallel for
		for (int i = first_second_internal_row; i <= last_second_internal_row; ++i) {
			set_cell(R, i, cell_cols - 2, -laplas_5(P, i, cell_cols - 2, P_neigh) - F(X(cell_cols - 2), Y(i)) - phi(X(cell_cols - 1), Y(i)) * coef_h_1);
		}		
	}

	send_borders(R);
	receive_borders(R_neigh, req);
	sync_receive_borders(req, 4);
}

static inline void
calculate()
{
	iterations_count++;

	/*Рассчитываем tau = (-laplas(r), r) / (-laplas(r), -laplas(r)). */
	laplas_5_matrix(R, R_neigh, matrix_buffer);
	double numerator = local_scalar(matrix_buffer, R);
	double denominator = local_scalar(matrix_buffer, matrix_buffer);
	double tau = global_scalar_fraction(numerator, denominator);

	/* Расчет погрешности. */
	double local_error, global_error;
	double local_inc = calculate_next_P(tau, R, R_neigh, &local_error,h_1 * h_2);
	double global_inc = global_increment(local_inc, local_error, &global_error, h_1 * h_2);
	if (proc_rank == 0)
		printf("global_increment = %lf\n", global_inc);

	MPI_Request req[4];
	for (int i = 0; i < 4; ++i)
		req[i] = MPI_REQUEST_NULL;

	while (global_inc > epsilon &&
	       iterations_count < max_iterations_count) {
	       	++iterations_count;
		calculate_next_R();
		send_borders(R);
		receive_borders(R_neigh, req);
		sync_receive_borders(req, 4);

		laplas_5_matrix(R, R_neigh, matrix_buffer);
		double numerator = local_scalar(matrix_buffer, R);
		double denominator = local_scalar(matrix_buffer, matrix_buffer);
		double tau = global_scalar_fraction(numerator, denominator);

		memcpy(P_prev, P, cell_rows * cell_cols);
		double local_inc = calculate_next_P(tau, R, R_neigh, &local_error, h_1 * h_2);
		global_inc = global_increment(local_inc, local_error, &global_error, h_1 * h_2);
		if (proc_rank == 0 && iterations_count % 10 == 0) {
			printf("global_increment = %lf, global_error = %lf\n", global_inc, global_error);
		}
	}
	printf("finished in %d iterations\n", iterations_count);
}

int
main(int argc, char **argv)
{
	int proc_count;
	int table_height, table_width;
	if (argc == 3) {
		table_height = atoi(argv[1]);
		if (table_height == 0) {
			printf("Incorrect table height\n");
			return -1;
		}
		table_width = atoi(argv[2]);
		if (table_width == 0) {
			printf("Incorrect table width\n");
			return -1;
		}
	} else {
		table_height = 500;
		table_width = 500;
	}

	h_1 = (A_2 - A_1) / (table_width - 1);   // table_width - 1 = M
	h_2 = (B_2 - B_1) / (table_height - 1);  // table_height - 1 = N
	coef_h_1 = 1 / (h_1 * h_1);
	coef_h_2 = 1 / (h_2 * h_2);
	MPI_Init(&argc, &argv);  // Инициализация MPI
	MPI_Comm_size(MPI_COMM_WORLD, &proc_count); //Получение общего числа процессов 
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);  //Получение порядкового номера процесса 

	if (proc_rank == 0) {                    //Нулевой процесс определяет количество потоков
                int num_of_threads = 0;
                #pragma omp parallel		 //Создание параллельного региона
                {
                        #pragma omp atomic       //Aтомарный доступ к определенной ячейке памяти.
                        num_of_threads++;
                }
                printf("Number of threads = %d\n", num_of_threads);
        }

	/**
 	* Распределение матрицы по процессам, вычисление процессорной сетки
	*/

	if (calculate_cells(table_height, table_width, proc_count) != 0) {
		printf("Cannot split table in a specified process count\n");
		goto error;
	}
	/**
 	* Выделение памяти под матрицы, заполенение матриц начальными значениями
	*/	
	create_matrices();

	start_time = MPI_Wtime();
	calculate();
	end_time = MPI_Wtime();
	if (proc_rank == 0)
		printf("time = %lf\n", end_time - start_time);
	free(P);
	for (int i = 0; i < 4; ++i)
		free(P_neigh[i]);

	MPI_Finalize();
	return 0;
error:
	MPI_Finalize();
	return -1;
}
