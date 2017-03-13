/*
 * general_function.c
 *
 *  Created on: 1 mars 2017
 *      Author: tepelbaum
 */



#include <math.h>
#include <omp.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>



#include "general_function.h"
#include "macro.h"
#include "structs.h"


//////////////////// mean vector START ////////////////////////////////
// require a vector and its size
double mean_func(double * x, int L) {
	double mean = 0;
	int i;
	for (i=0;i<L;i++) {
		mean+=x[i];
	}
  return mean/(double)L;
}
//////////////////// mean vector END //////////////////////////////////



//////////////////// sum vector START /////////////////////////////////
// require a vector and its size
double sum_func(double * x, int L) {
	double sum = 0;
	int i;
	for (i=0;i<L;i++) {
		sum+=x[i];
	}
  return sum;
}
//////////////////// sum vector END //////////////////////////////////



//////////////////// sum exp vector START ////////////////////////////
// require a vector and its size
double sumexp_func(double * x, int L) {
	double sumexp = 0;
	int i;
	for (i=0;i<L;i++) {
		sumexp+=exp(x[i]);
	}
  return sumexp;
}
//////////////////// sum exp vector END //////////////////////////////



//////////////////// biased std vector START /////////////////////////
// require a vector and its size
double biased_std_func(double * x, int L) {
	double mean = mean_func(x,L);
	double std = 0;
	int i;
	for (i=0;i<L;i++) {
		std+=x[i]*x[i];
	}
  return sqrt(std/(double)L-mean*mean);
}
//////////////////// biased std vector END ///////////////////////////



////////////////////  unbiased std vector START //////////////////////
// require a vector and its size
double unbiased_std_func(double * x, int L) {
	double mean = mean_func(x,L);
	double std = 0;
	int i;
	for (i=0;i<L;i++) {
		std+=(x[i]-mean)*(x[i]-mean);
	}
  return sqrt(std/(double)(L-1));
}
////////////////////  unbiased std vector END ////////////////////////
