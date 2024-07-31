///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    real.h
// Description: Header file for real matrix multiplication
//
// Note:        DO NOT MODIFY THIS CODE!
///////////////////////////////////////////////////////////////////////////////

#ifndef __REAL_H__
#define __REAL_H__

#include <stdio.h>
#include <stdlib.h>

#include <ap_int.h>

typedef ap_int<16> real_t;

#define MATRIX_N 150
#define MATRIX_M 100
#define MATRIX_K 200

void real_matmul( 
    real_t MatA_DRAM[MATRIX_M][MATRIX_N], 
    real_t MatB_DRAM[MATRIX_N][MATRIX_K], 
    real_t MatC_DRAM[MATRIX_M][MATRIX_K]
);

#endif
