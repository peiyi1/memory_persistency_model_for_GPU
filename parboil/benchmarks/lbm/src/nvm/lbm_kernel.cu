/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef LBM_KERNEL_CU
#define LBM_KERNEL_CU

#include "lbm.h"
#include "nvm_util.h"

__device__ float NVM_log[LOG_SIZE_16M];
__device__ int NVM_flag[FLAG_SIZE_1M];
#include "nvmg.cu"
#include "nvmi.cu"
#include "nvm3.cu"
#include "nvm4.cu"
#include "nvm5.cu"
#include "nvm6.cu"
/******************************************************************************/

__global__ void performStreamCollide_kernel( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  DST_C ( dstGrid ) = tempC;

  DST_N ( dstGrid ) = tempN; 
  DST_S ( dstGrid ) = tempS;
  DST_E ( dstGrid ) = tempE;
  DST_W ( dstGrid ) = tempW;
  DST_T ( dstGrid ) = tempT;
  DST_B ( dstGrid ) = tempB;

  DST_NE( dstGrid ) = tempNE;
  DST_NW( dstGrid ) = tempNW;
  DST_SE( dstGrid ) = tempSE;
  DST_SW( dstGrid ) = tempSW;
  DST_NT( dstGrid ) = tempNT;
  DST_NB( dstGrid ) = tempNB;
  DST_ST( dstGrid ) = tempST;
  DST_SB( dstGrid ) = tempSB;
  DST_ET( dstGrid ) = tempET;
  DST_EB( dstGrid ) = tempEB;
  DST_WT( dstGrid ) = tempWT;
  DST_WB( dstGrid ) = tempWB;
}

__global__ void performStreamCollide_kernel_nvmo( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  DST_C ( dstGrid ) = tempC;
  CLFLUSH(&DST_C(dstGrid));
  MEM_FENCE;
  DST_N ( dstGrid ) = tempN; 
  CLFLUSH(&DST_N(dstGrid));
  MEM_FENCE;
  DST_S ( dstGrid ) = tempS;
  CLFLUSH(&DST_S(dstGrid));
  MEM_FENCE;
  DST_E ( dstGrid ) = tempE;
  CLFLUSH(&DST_E(dstGrid));
  MEM_FENCE;
  DST_W ( dstGrid ) = tempW;
  CLFLUSH(&DST_W(dstGrid));
  MEM_FENCE;
  DST_T ( dstGrid ) = tempT;
  CLFLUSH(&DST_T(dstGrid));
  MEM_FENCE;
  DST_B ( dstGrid ) = tempB;
  CLFLUSH(&DST_B(dstGrid));
  MEM_FENCE;

  DST_NE( dstGrid ) = tempNE;
  CLFLUSH(&DST_NE(dstGrid));
  MEM_FENCE;
  DST_NW( dstGrid ) = tempNW;
  CLFLUSH(&DST_NW(dstGrid));
  MEM_FENCE;
  DST_SE( dstGrid ) = tempSE;
  CLFLUSH(&DST_SE(dstGrid));
  MEM_FENCE;
  DST_SW( dstGrid ) = tempSW;
  CLFLUSH(&DST_SW(dstGrid));
  MEM_FENCE;
  DST_NT( dstGrid ) = tempNT;
  CLFLUSH(&DST_NT(dstGrid));
  MEM_FENCE;
  DST_NB( dstGrid ) = tempNB;
  CLFLUSH(&DST_NB(dstGrid));
  MEM_FENCE;
  DST_ST( dstGrid ) = tempST;
  CLFLUSH(&DST_ST(dstGrid));
  MEM_FENCE;
  DST_SB( dstGrid ) = tempSB;
  CLFLUSH(&DST_SB(dstGrid));
  MEM_FENCE;
  DST_ET( dstGrid ) = tempET;
  CLFLUSH(&DST_ET(dstGrid));
  MEM_FENCE;
  DST_EB( dstGrid ) = tempEB;
  CLFLUSH(&DST_EB(dstGrid));
  MEM_FENCE;
  DST_WT( dstGrid ) = tempWT;
  CLFLUSH(&DST_WT(dstGrid));
  MEM_FENCE;
  DST_WB( dstGrid ) = tempWB;
  CLFLUSH(&DST_WB(dstGrid));
  MEM_FENCE;
}



__global__ void performStreamCollide_kernel_nvmw( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  DST_C ( dstGrid ) = tempC;
  DST_N ( dstGrid ) = tempN; 
  DST_S ( dstGrid ) = tempS;
  DST_E ( dstGrid ) = tempE;
  DST_W ( dstGrid ) = tempW;
  DST_T ( dstGrid ) = tempT;
  DST_B ( dstGrid ) = tempB;

  DST_NE( dstGrid ) = tempNE;
  DST_NW( dstGrid ) = tempNW;
  DST_SE( dstGrid ) = tempSE;
  DST_SW( dstGrid ) = tempSW;
  DST_NT( dstGrid ) = tempNT;
  DST_NB( dstGrid ) = tempNB;
  DST_ST( dstGrid ) = tempST;
  DST_SB( dstGrid ) = tempSB;
  DST_ET( dstGrid ) = tempET;
  DST_EB( dstGrid ) = tempEB;
  DST_WT( dstGrid ) = tempWT;
  DST_WB( dstGrid ) = tempWB;

  CLFLUSH(&DST_C(dstGrid));
  CLFLUSH(&DST_N(dstGrid));
  CLFLUSH(&DST_S(dstGrid));
  CLFLUSH(&DST_E(dstGrid));
  CLFLUSH(&DST_W(dstGrid));
  CLFLUSH(&DST_T(dstGrid));
  CLFLUSH(&DST_B(dstGrid));
  CLFLUSH(&DST_NE(dstGrid));
  CLFLUSH(&DST_NW(dstGrid));
  CLFLUSH(&DST_SE(dstGrid));
  CLFLUSH(&DST_SW(dstGrid));
  CLFLUSH(&DST_NT(dstGrid));
  CLFLUSH(&DST_NB(dstGrid));
  CLFLUSH(&DST_ST(dstGrid));
  CLFLUSH(&DST_SB(dstGrid));
  CLFLUSH(&DST_ET(dstGrid));
  CLFLUSH(&DST_EB(dstGrid));
  CLFLUSH(&DST_WT(dstGrid));
  CLFLUSH(&DST_WB(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
}


__global__ void performStreamCollide_kernel_nvmq( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  DST_C ( dstGrid ) = tempC;
  DST_N ( dstGrid ) = tempN; 
  DST_S ( dstGrid ) = tempS;
  DST_E ( dstGrid ) = tempE;
  DST_W ( dstGrid ) = tempW;
  DST_T ( dstGrid ) = tempT;
  DST_B ( dstGrid ) = tempB;

  DST_NE( dstGrid ) = tempNE;
  DST_NW( dstGrid ) = tempNW;
  DST_SE( dstGrid ) = tempSE;
  DST_SW( dstGrid ) = tempSW;
  DST_NT( dstGrid ) = tempNT;
  DST_NB( dstGrid ) = tempNB;
  DST_ST( dstGrid ) = tempST;
  DST_SB( dstGrid ) = tempSB;
  DST_ET( dstGrid ) = tempET;
  DST_EB( dstGrid ) = tempEB;
  DST_WT( dstGrid ) = tempWT;
  DST_WB( dstGrid ) = tempWB;

  CLFLUSH(&DST_C(dstGrid));
  CLFLUSH(&DST_N(dstGrid));
  CLFLUSH(&DST_S(dstGrid));
  CLFLUSH(&DST_E(dstGrid));
  CLFLUSH(&DST_W(dstGrid));
  CLFLUSH(&DST_T(dstGrid));
  CLFLUSH(&DST_B(dstGrid));
  CLFLUSH(&DST_NE(dstGrid));
  CLFLUSH(&DST_NW(dstGrid));
  CLFLUSH(&DST_SE(dstGrid));
  CLFLUSH(&DST_SW(dstGrid));
  CLFLUSH(&DST_NT(dstGrid));
  CLFLUSH(&DST_NB(dstGrid));
  CLFLUSH(&DST_ST(dstGrid));
  CLFLUSH(&DST_SB(dstGrid));
  CLFLUSH(&DST_ET(dstGrid));
  CLFLUSH(&DST_EB(dstGrid));
  CLFLUSH(&DST_WT(dstGrid));
  CLFLUSH(&DST_WB(dstGrid));
  MEM_FENCE;
}


__global__ void performStreamCollide_kernel_nvmu( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  DST_C ( dstGrid ) = tempC;
  CLFLUSH(&DST_C(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_N ( dstGrid ) = tempN; 
  CLFLUSH(&DST_N(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_S ( dstGrid ) = tempS;
  CLFLUSH(&DST_S(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_E ( dstGrid ) = tempE;
  CLFLUSH(&DST_E(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_W ( dstGrid ) = tempW;
  CLFLUSH(&DST_W(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_T ( dstGrid ) = tempT;
  CLFLUSH(&DST_T(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_B ( dstGrid ) = tempB;
  CLFLUSH(&DST_B(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;

  DST_NE( dstGrid ) = tempNE;
  CLFLUSH(&DST_NE(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_NW( dstGrid ) = tempNW;
  CLFLUSH(&DST_NW(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_SE( dstGrid ) = tempSE;
  CLFLUSH(&DST_SE(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_SW( dstGrid ) = tempSW;
  CLFLUSH(&DST_SW(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_NT( dstGrid ) = tempNT;
  CLFLUSH(&DST_NT(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_NB( dstGrid ) = tempNB;
  CLFLUSH(&DST_NB(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_ST( dstGrid ) = tempST;
  CLFLUSH(&DST_ST(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_SB( dstGrid ) = tempSB;
  CLFLUSH(&DST_SB(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_ET( dstGrid ) = tempET;
  CLFLUSH(&DST_ET(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_EB( dstGrid ) = tempEB;
  CLFLUSH(&DST_EB(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_WT( dstGrid ) = tempWT;
  CLFLUSH(&DST_WT(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  DST_WB( dstGrid ) = tempWB;
  CLFLUSH(&DST_WB(dstGrid));
  MEM_FENCE; PCOMMIT; MEM_FENCE;
}


__global__ void performStreamCollide_kernel_nvmb( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  ST_WT_FLOAT(&DST_C(dstGrid), tempC);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_N(dstGrid), tempN);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_S(dstGrid), tempS);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_E(dstGrid), tempE);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_W(dstGrid), tempW);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_T(dstGrid), tempT);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_B(dstGrid), tempB);
  MEM_FENCE;

  ST_WT_FLOAT(&DST_NE(dstGrid), tempNE);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_NW(dstGrid), tempNW);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_SE(dstGrid), tempSE);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_SW(dstGrid), tempSW);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_NT(dstGrid), tempNT);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_NB(dstGrid), tempNB);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_ST(dstGrid), tempST);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_SB(dstGrid), tempSB);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_ET(dstGrid), tempET);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_EB(dstGrid), tempEB);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_WT(dstGrid), tempWT);
  MEM_FENCE;
  ST_WT_FLOAT(&DST_WB(dstGrid), tempWB);
  MEM_FENCE;
}


__global__ void performStreamCollide_kernel_nvmj( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  ST_WT_FLOAT(&DST_C(dstGrid), tempC);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_N(dstGrid), tempN);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_S(dstGrid), tempS);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_E(dstGrid), tempE);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_W(dstGrid), tempW);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_T(dstGrid), tempT);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_B(dstGrid), tempB);
  MEM_FENCE; PCOMMIT; MEM_FENCE;

  ST_WT_FLOAT(&DST_NE(dstGrid), tempNE);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_NW(dstGrid), tempNW);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_SE(dstGrid), tempSE);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_SW(dstGrid), tempSW);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_NT(dstGrid), tempNT);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_NB(dstGrid), tempNB);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_ST(dstGrid), tempST);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_SB(dstGrid), tempSB);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_ET(dstGrid), tempET);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_EB(dstGrid), tempEB);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_WT(dstGrid), tempWT);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
  ST_WT_FLOAT(&DST_WB(dstGrid), tempWB);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
}

__global__ void performStreamCollide_kernel_nvmd( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  ST_WT_FLOAT(&DST_C(dstGrid), tempC);
  ST_WT_FLOAT(&DST_N(dstGrid), tempN);
  ST_WT_FLOAT(&DST_S(dstGrid), tempS);
  ST_WT_FLOAT(&DST_E(dstGrid), tempE);
  ST_WT_FLOAT(&DST_W(dstGrid), tempW);
  ST_WT_FLOAT(&DST_T(dstGrid), tempT);
  ST_WT_FLOAT(&DST_B(dstGrid), tempB);

  ST_WT_FLOAT(&DST_NE(dstGrid), tempNE);
  ST_WT_FLOAT(&DST_NW(dstGrid), tempNW);
  ST_WT_FLOAT(&DST_SE(dstGrid), tempSE);
  ST_WT_FLOAT(&DST_SW(dstGrid), tempSW);
  ST_WT_FLOAT(&DST_NT(dstGrid), tempNT);
  ST_WT_FLOAT(&DST_NB(dstGrid), tempNB);
  ST_WT_FLOAT(&DST_ST(dstGrid), tempST);
  ST_WT_FLOAT(&DST_SB(dstGrid), tempSB);
  ST_WT_FLOAT(&DST_ET(dstGrid), tempET);
  ST_WT_FLOAT(&DST_EB(dstGrid), tempEB);
  ST_WT_FLOAT(&DST_WT(dstGrid), tempWT);
  ST_WT_FLOAT(&DST_WB(dstGrid), tempWB);
  MEM_FENCE;
}


__global__ void performStreamCollide_kernel_nvml( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  ST_WT_FLOAT(&DST_C(dstGrid), tempC);
  ST_WT_FLOAT(&DST_N(dstGrid), tempN);
  ST_WT_FLOAT(&DST_S(dstGrid), tempS);
  ST_WT_FLOAT(&DST_E(dstGrid), tempE);
  ST_WT_FLOAT(&DST_W(dstGrid), tempW);
  ST_WT_FLOAT(&DST_T(dstGrid), tempT);
  ST_WT_FLOAT(&DST_B(dstGrid), tempB);

  ST_WT_FLOAT(&DST_NE(dstGrid), tempNE);
  ST_WT_FLOAT(&DST_NW(dstGrid), tempNW);
  ST_WT_FLOAT(&DST_SE(dstGrid), tempSE);
  ST_WT_FLOAT(&DST_SW(dstGrid), tempSW);
  ST_WT_FLOAT(&DST_NT(dstGrid), tempNT);
  ST_WT_FLOAT(&DST_NB(dstGrid), tempNB);
  ST_WT_FLOAT(&DST_ST(dstGrid), tempST);
  ST_WT_FLOAT(&DST_SB(dstGrid), tempSB);
  ST_WT_FLOAT(&DST_ET(dstGrid), tempET);
  ST_WT_FLOAT(&DST_EB(dstGrid), tempEB);
  ST_WT_FLOAT(&DST_WT(dstGrid), tempWT);
  ST_WT_FLOAT(&DST_WB(dstGrid), tempWB);
  MEM_FENCE; PCOMMIT; MEM_FENCE;
}



__global__ void performStreamCollide_kernel_nvm1( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  DST_C ( dstGrid ) = tempC;
  DST_N ( dstGrid ) = tempN; 
  DST_S ( dstGrid ) = tempS;
  DST_E ( dstGrid ) = tempE;
  DST_W ( dstGrid ) = tempW;
  DST_T ( dstGrid ) = tempT;
  DST_B ( dstGrid ) = tempB;

  DST_NE( dstGrid ) = tempNE;
  DST_NW( dstGrid ) = tempNW;
  DST_SE( dstGrid ) = tempSE;
  DST_SW( dstGrid ) = tempSW;
  DST_NT( dstGrid ) = tempNT;
  DST_NB( dstGrid ) = tempNB;
  DST_ST( dstGrid ) = tempST;
  DST_SB( dstGrid ) = tempSB;
  DST_ET( dstGrid ) = tempET;
  DST_EB( dstGrid ) = tempEB;
  DST_WT( dstGrid ) = tempWT;
  DST_WB( dstGrid ) = tempWB;

  L2WB;
  MEM_FENCE;
}

__global__ void performStreamCollide_kernel_nvm2( float* srcGrid, float* dstGrid ) 
{

  //Using some predefined macros here.  Consider this the declaration 
  //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

  //Load all of the input fields
  //This is a gather operation of the SCATTER preprocessor variable
  // is undefined in layout_config.h, or a "local" read otherwise
  tempC = SRC_C(srcGrid);
  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);
  tempNE= SRC_NE(srcGrid);
  tempNW= SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  //Test whether the cell is fluid or obstacle
  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
    //Swizzle the inputs: reflect any fluid coming into this cell 
    // back to where it came from
    temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
    temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
    temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
    temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
    temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
    temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
    temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
  }
  else {
    //The math meat of LBM: ignore for optimization
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
      + tempS + tempE
      + tempW + tempT
      + tempB + tempNE
      + tempNW + tempSE
      + tempSW + tempNT
      + tempNB + tempST
      + tempSB + tempET
      + tempEB + tempWT
      + tempWB;

    ux = + tempE - tempW
      + tempNE - tempNW
      + tempSE - tempSW
      + tempET + tempEB
      - tempWT - tempWB;
    uy = + tempN - tempS
      + tempNE + tempNW
      - tempSE - tempSW
      + tempNT + tempNB
      - tempST - tempSB;
    uz = + tempT - tempB
      + tempNT - tempNB
      + tempST - tempSB
      + tempET - tempEB
      + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;
    if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }
    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;


    //Put the output values for this cell in the shared memory
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;	
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  //Write the results computed above
  //This is a scatter operation of the SCATTER preprocessor variable
  // is defined in layout_config.h, or a "local" write otherwise
  DST_C ( dstGrid ) = tempC;
  DST_N ( dstGrid ) = tempN; 
  DST_S ( dstGrid ) = tempS;
  DST_E ( dstGrid ) = tempE;
  DST_W ( dstGrid ) = tempW;
  DST_T ( dstGrid ) = tempT;
  DST_B ( dstGrid ) = tempB;

  DST_NE( dstGrid ) = tempNE;
  DST_NW( dstGrid ) = tempNW;
  DST_SE( dstGrid ) = tempSE;
  DST_SW( dstGrid ) = tempSW;
  DST_NT( dstGrid ) = tempNT;
  DST_NB( dstGrid ) = tempNB;
  DST_ST( dstGrid ) = tempST;
  DST_SB( dstGrid ) = tempSB;
  DST_ET( dstGrid ) = tempET;
  DST_EB( dstGrid ) = tempEB;
  DST_WT( dstGrid ) = tempWT;
  DST_WB( dstGrid ) = tempWB;

  L2WB;
  MEM_FENCE; PCOMMIT; MEM_FENCE;
}

#endif // LBM_KERNEL_CU
