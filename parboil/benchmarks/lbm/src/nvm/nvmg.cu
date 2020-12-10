
__global__ void performStreamCollide_kernel_nvmg( float* srcGrid, float* dstGrid ) 
{
  SWEEP_VAR
    SWEEP_X = threadIdx.x;
  SWEEP_Y = blockIdx.x;
  SWEEP_Z = blockIdx.y;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
  float tempSB, tempET, tempEB, tempWT, tempWB ;

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

  if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
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

  // Create undo logs
  ST_WT_FLOAT(&DST_C(NVM_log), tempC);
  ST_WT_FLOAT(&DST_N(NVM_log), tempN);
  ST_WT_FLOAT(&DST_S(NVM_log), tempS);
  ST_WT_FLOAT(&DST_E(NVM_log), tempE);
  ST_WT_FLOAT(&DST_W(NVM_log), tempW);
  ST_WT_FLOAT(&DST_T(NVM_log), tempT);
  ST_WT_FLOAT(&DST_B(NVM_log), tempB);
  ST_WT_FLOAT(&DST_NE(NVM_log), tempNE);
  ST_WT_FLOAT(&DST_NW(NVM_log), tempNW);
  ST_WT_FLOAT(&DST_SE(NVM_log), tempSE);
  ST_WT_FLOAT(&DST_SW(NVM_log), tempSW);
  ST_WT_FLOAT(&DST_NT(NVM_log), tempNT);
  ST_WT_FLOAT(&DST_NB(NVM_log), tempNB);
  ST_WT_FLOAT(&DST_ST(NVM_log), tempST);
  ST_WT_FLOAT(&DST_SB(NVM_log), tempSB);
  ST_WT_FLOAT(&DST_ET(NVM_log), tempET);
  ST_WT_FLOAT(&DST_EB(NVM_log), tempEB);
  ST_WT_FLOAT(&DST_WT(NVM_log), tempWT);
  ST_WT_FLOAT(&DST_WB(NVM_log), tempWB);
  MEM_FENCE;
  SET_NVM_FLAG(1);
  __syncthreads();


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
  __syncthreads();
  SET_NVM_FLAG(2);
}