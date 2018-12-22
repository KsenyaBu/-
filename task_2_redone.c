#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static double *P = NULL;
static double *R = NULL;
static double *matrix_temp = NULL;
static double *matrix_buffer = NULL;
static double *matrix_buffer1 = NULL;
static double *matrix_buffer2 = NULL;
static double *F_matrix = NULL;
static double *old = NULL;
static const double A1 = -1;
static const double A2 = 2;
static const double B1 = -2;
static const double B2 = 2;
static int cols = 20;             //число точек сетки по X
static int rows = 20;             //число точек сетки по Y
static double cur_error = 1;
static double error = 0.000001;
static double x_step;
static double y_step;

static inline double cell(double* matrix, int col_x, int row_y){
    return matrix[row_y * cols + col_x];
}

static inline void set_cell(double* matrix, int col_x, int row_y, double value){
    matrix[row_y * cols + col_x] = value;
}

static inline double X(int i){
    return A1 + i * x_step;
}

static inline double Y(int j){
    return B1 + j * y_step;
}

static inline double F(double x, double y){
    double tmp = (x + y)*(x + y);
    return -4 * (2*tmp - 1) * exp(1-tmp);
}

static inline double phi(double x, double y){
    return exp(1 - (x + y)*(x + y));
}

static inline double scalar(double* a, double* b){
    double ret = 0;
    double temp = 0;
    int i,j;
    for (j = 0; j <= rows-1; j++)
    {
        for (i = 0; i <= cols-1; i++)
        {
            if (i == 0 || j == 0 || i == cols-1 || j == rows-1)
                temp = 0;
            else
                temp = 1;
            ret += temp * cell(a, i, j) * cell(b, i, j) * (x_step * y_step);
        }
    }
    return ret;
}

static inline void laplas_5_matrix(double* matrix)
{
    double a11, a21, a01, a10, a12;
    double tmp1, tmp2;
    int i,j;
    for (j = 2; j <= rows-3; j++)
    {
        for (i = 2; i <= cols-3; i++)
        {
            a11 = cell(matrix, i, j);
            a21 = cell(matrix, i + 1, j);
            a01 = cell(matrix, i - 1, j);
            a10 = cell(matrix, i, j - 1);
            a12 = cell(matrix, i, j + 1);
            tmp1 = (2*a11 - a01 - a21)/(x_step * x_step);
            tmp2 = (2*a11 - a10 - a12)/(y_step * y_step);
            set_cell(matrix_temp, i, j, (tmp1 + tmp2));
        }
    }

    for (j = 2; j <= rows-3; j++)
    {
        a11 = cell(matrix, 1, j);
        a21 = cell(matrix, 2, j);
        a10 = cell(matrix, 1, j - 1);
        a12 = cell(matrix, 1, j + 1);
        tmp1 = (2*a11 - a21)/(x_step * x_step);
        tmp2 = (2*a11 - a10 - a12)/(y_step * y_step);
        set_cell(matrix_temp, 1, j, (tmp1 + tmp2));
    }

    for (j = 2; j <= rows-3; j++)
    {
        a11 = cell(matrix, cols - 2, j);
        a01 = cell(matrix, cols - 3, j);
        a10 = cell(matrix, cols - 2, j - 1);
        a12 = cell(matrix, cols - 2, j + 1);
        tmp1 = (2*a11 - a01)/(x_step * x_step);
        tmp2 = (2*a11 - a10 - a12)/(y_step * y_step);
        set_cell(matrix_temp, cols - 2, j, (tmp1 + tmp2));
    }

    for (i = 2; i <= cols-3; i++)
    {
        a11 = cell(matrix, i, 1);
        a01 = cell(matrix, i+1, 1);
        a21 = cell(matrix, i-1, 1);
        a12 = cell(matrix, i, 2);
        tmp1 = (2*a11 - a01 - a21)/(x_step * x_step);
        tmp2 = (2*a11 - a12)/(y_step * y_step);
        set_cell(matrix_temp, i, 1, (tmp1 + tmp2));
    }

    for (i = 2; i <= cols-3; i++)
    {
        a11 = cell(matrix, i, rows -2);
        a01 = cell(matrix, i+1, rows - 2);
        a21 = cell(matrix, i-1, rows - 2);
        a10 = cell(matrix, i, rows - 3);
        tmp1 = (2*a11 - a01 - a21)/(x_step * x_step);
        tmp2 = (2*a11 - a10)/(y_step * y_step);
        set_cell(matrix_temp, i, rows - 2, (tmp1 + tmp2));
    }

    a11 = cell(matrix, 1, 1);
    a21 = cell(matrix, 2, 1);
    a12 = cell(matrix, 1, 2);
    tmp1 = (2*a11 - a21)/(x_step * x_step);
    tmp2 = (2*a11 - a12)/(y_step * y_step);
    set_cell(matrix_temp, 1, 1, (tmp1 + tmp2));

    a11 = cell(matrix, cols-2, 1);
    a01 = cell(matrix, cols-3, 1);
    a12 = cell(matrix, cols-2, 2);
    tmp1 = (2*a11 - a01)/(x_step * x_step);
    tmp2 = (2*a11 - a12)/(y_step * y_step);
    set_cell(matrix_temp, cols-2, 1, (tmp1 + tmp2));

    a11 = cell(matrix, 1, rows - 2);
    a21 = cell(matrix, 2, rows - 2);
    a10 = cell(matrix, 1, rows - 3);
    tmp1 = (2*a11 - a21)/(x_step * x_step);
    tmp2 = (2*a11 - a10)/(y_step * y_step);
    set_cell(matrix_temp, 1, rows - 2, (tmp1 + tmp2));

    a11 = cell(matrix, cols - 2, rows - 2);
    a01 = cell(matrix, cols - 3, rows - 2);
    a10 = cell(matrix, cols - 2, rows - 3);
    tmp1 = (2*a11 - a01)/(x_step * x_step);
    tmp2 = (2*a11 - a10)/(y_step * y_step);
    set_cell(matrix_temp, cols-2, rows - 1, (tmp1 + tmp2));

    for (j = 0; j <= rows-1; j++)
    {
        set_cell(matrix_temp, 0, j, 0);
        set_cell(matrix_temp, cols-1, j, 0);
    }

    for (i = 0; j <= cols-1; j++)
    {
        set_cell(matrix_temp, i, 0, 0);
        set_cell(matrix_temp, i, rows-1, 0);
    }
    return;
}

static inline void calculate_next_R(FILE *f)
{
    laplas_5_matrix(P);
    int i,j;
    for (j = 0; j <= rows-1; j++)
        for (i = 0; i <= cols-1; i++)
            set_cell(R, i, j, cell(matrix_temp, i, j) - cell(F_matrix,i,j));
    return;
}

static inline void calculate_next_P(double tau){
    double old_P;
    int i,j;
    for (j = 0; j <= rows-1; j++)
    {
        for (i = 0; i <= cols-1; i++)
        {
            old_P = cell(P, i, j);
            set_cell(P, i, j, old_P - tau * cell(R, i, j));
        }
    }
}

int main(int argc, char ** argv)
{
    char filename[] = "ksy_20.txt";
    FILE *f = fopen(filename, "w");
	if (f == NULL)
	{
		printf("Error!\n");
		exit(1);
	}
    double *mem = (double *) calloc(rows * cols * 8, sizeof(double));
	P = mem;
	R = mem + rows * cols;
	matrix_temp = mem + rows * cols * 2;
    matrix_buffer = mem + rows * cols * 3;
	matrix_buffer1 = mem + rows * cols * 4;
	matrix_buffer2 = mem + rows * cols * 5;
    old = mem + rows * cols * 6;
    F_matrix = mem + rows * cols * 7;
    x_step = (A2 - A1)/(cols - 1);    //M = cols - 1
    y_step = (B2 - B1)/(rows - 1);    //N = rows - 1
    fprintf(f,"x_step = %f\t", x_step);
    fprintf(f,"y_step = %f\n", y_step);
    double matrix_exact[cols * rows];
    int j = 0;
    int i = 0;
    fprintf(f,"F\n");
    for (j = 2; j <= rows-3; j++)
        for (i = 2; i <= cols-3; i++)
            set_cell(F_matrix, i, j, F(X(i),Y(j)));

    for (j = 2; j <= rows-3; j++)
    {
        set_cell(F_matrix, 1, j, F(X(1), Y(j)) + phi(X(0), Y(j))/(x_step * x_step));
        set_cell(F_matrix, cols - 2, j, F(X(cols - 2), Y(j)) + phi(X(cols - 1), Y(j))/(x_step * x_step));
    }

    for (i = 2; i <= cols-3; i++)
    {
        set_cell(F_matrix, i, 1, F(X(i), Y(1)) + phi(X(i), Y(0))/(y_step * y_step));
        set_cell(F_matrix, i, rows-2, F(X(i), Y(rows-2)) + phi(X(i), Y(rows-1))/(y_step * y_step));
    }

    set_cell(F_matrix, 1, 1, F(X(1), Y(1)) + phi(X(1), Y(0))/(y_step * y_step) + phi(X(0), Y(1))/(x_step * x_step));
    set_cell(F_matrix, 1, rows - 2, F(X(1), Y(rows-2)) + phi(X(1), Y(rows-1))/(y_step * y_step) + phi(X(0), Y(rows-2))/(x_step * x_step));
    set_cell(F_matrix, cols - 2, 1, F(X(cols - 2), Y(1)) + phi(X(cols-2), Y(0))/(y_step * y_step) + phi(X(cols-1), Y(1))/(x_step * x_step));
    set_cell(F_matrix, i, j, F(X(cols-2), Y(rows-2)) + phi(X(cols-2), Y(rows-1))/(y_step * y_step) + phi(X(cols-1), Y(rows-2))/(x_step * x_step));
/*    fprintf(f,"exact\n");
    for (j = 0; j <= rows-1; j++)
    {
        for (i = 0; i <= cols-1; i++)
        {
            set_cell(matrix_exact, i, j, phi(X(i),Y(j)));
            fprintf(f,"%f\t", cell(matrix_exact,i,j));
        }
        fprintf(f,"\n");
    }

    fprintf(f,"\nF\n");
    for (j = 0; j <= rows-1; j++)
    {
        for (i = 0; i <= cols-1; i++)
        {
            fprintf(f,"%f\t", cell(F_matrix,i,j));
        }
        fprintf(f,"\n");
    }*/

    for (j = 0; j <= rows-1; j++)
    {
        for (i = 0; i <= cols-1; i++)
        {
            set_cell(P, i, j, 0);
            set_cell(R, i, j, 0);
        }
    }

    for (i = 0; i <= cols-1; i++)
    {
        set_cell(P, i, 0, phi(X(i), Y(0)));
        set_cell(P, i, rows - 1, phi(X(i), Y(rows - 1)));
    }

    for (j = 0; j <= rows-1; j++)
    {
        set_cell(P, 0, j, phi(X(0), Y(j)));
        set_cell(P, cols - 1, j , phi(X(cols - 1), Y(j)));
    }

    calculate_next_R(f);

    int count = 1;
    double tmp1, tmp2, tau;
    double global_error = 0;
    while (cur_error >= error)
    {
        fprintf(f,"%d\n",count);
        for (j = 0; j <= rows-1; j++)
            for (i = 0; i <= cols-1; i++)
                set_cell(old, i, j, cell(P, i, j));
        laplas_5_matrix(R);
        tmp1 = scalar(matrix_temp, R);
        tmp2 = scalar(matrix_temp, matrix_temp);
        tau = tmp1 / tmp2;
        calculate_next_P(tau);
        for (j = 0; j <= rows-1; j++)
        {
            for (i = 0; i <= cols-1; i++)
            {
                set_cell(matrix_buffer1, i, j, phi(X(i), Y(j)) - cell(P, i, j));
                set_cell(matrix_buffer2, i, j, cell(P, i, j) - cell(old, i, j));
             // fprintf(f,"%f\n",cell(P,i,j));
            }
        }
        cur_error = sqrt(scalar(matrix_buffer2,matrix_buffer2));
        global_error = sqrt(scalar(matrix_buffer1, matrix_buffer1));
        calculate_next_R(f);
        count += 1;
        fprintf(f,"error = %e\n",error);
        fprintf(f,"cur_error = %e\n",cur_error);
        fprintf(f,"%e\n",global_error);
    }
	return 0;
}
