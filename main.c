/*
 * main.c
 *
 *  Created on: 2 mars 2017
 *      Author: tepelbaum
 */


#include <math.h>
#include <omp.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>

#include "macro.h"
#include "structs.h"
#include "nn_spec.h"
#include "file_treatment.h"
#include "general_function.h"
#include "trafic_specific.h"
#include <sys\timeb.h>

///////// Parameters that control the evolution ///////////////////////
//double 	tin		= 1.0;
///////////////////////////////////////////////////////////////////////

int main (int argc,char *argv[])
{
	// Name of files, important variables
	int i,j,k,l;
	struct timeb start, end;
    int diff;
    ftime(&start);
	int iter_max = 50001;
	time_t t;
	srand((unsigned)time(&t));
	//srand (time (NULL));
	char speed_name[25] = "speedavg_20160";
	char * link_name = "links_periph_ordered.csv";
	double J_train, J_cv;
	int batch_size = 50;
	int * index = (int*)malloc(batch_size*sizeof(int));

	// Init the Neural Network
	Neural_Network *NN = (Neural_Network*)malloc(sizeof(Neural_Network));

	Init_NN(NN,speed_name,link_name);
	// momentum gradient, useful for momentum method
	double ** v_mom= (double **)malloc((L_layer-1)*sizeof (double));
	double ** v_mom_2= (double **)malloc((L_layer-1)*sizeof (double));
	for (i= 0;i<L_layer-1;i++) {
		v_mom[i] = (double *)malloc((NN->s_l[i+1])*(NN->s_l[i]+1)*sizeof (double));
		v_mom_2[i] = (double *)malloc((NN->s_l[i+1])*(NN->s_l[i]+1)*sizeof (double));
		Init_v_mom(v_mom[i],NN->s_l[i],NN->s_l[i+1]);
		Init_v_mom(v_mom_2[i],NN->s_l[i],NN->s_l[i+1]);
		printf("momentum between layer %d and %d initialized, matrix of dimension %d X %d \n",i,i+1,NN->s_l[i+1],NN->s_l[i]+1);
	}
	//

	// For dropout
	int ** mask= (int **)malloc((L_layer-1)*sizeof (int));
	for (i =0;i<L_layer-1;i++) {
		mask[i]= (int *)malloc(batch_size*(NN->s_l[i])*sizeof (int));
	}
	double mask_FP = 1;//0.5;//0.5;
	double mask_VU = 1;//0.8;//0.5;
	//

	// for NAG
	int gamma_coef = GAMMA;

	printf("Initialization done\n\n");



	for (i=0;i<iter_max;i++) {


		// Set mini batch indices
		for (j =0;j<batch_size;j++) {
			index[j]=(int) (round(drand48()*size_train_file));
		}
		//

		// Set mini batch masks
		for (j =0;j<batch_size;j++) {
			for (k =0;k<NN->s_l[0];k++) {
				mask[0][k+(NN->s_l[0])*j] = 1;//fmin(1,5*round(drand48())) ;
			}
		}
		for (l =1;l<L_layer-1;l++) {
			for (j =0;j<batch_size;j++) {
				for (k =0;k<NN->s_l[l];k++) {
					mask[l][k+(NN->s_l[l])*j] = 1;//round(drand48()) ;
				}
			}
		}
		//

		// FP mini batch
		FP_MBatch_drop(NN->a_train,NN->h_train,NN->s_l,NN->Theta,relu,c_output,batch_size,index,mask);
		//

		// BP mini batch
		//BP_MBatch_SGD_NAG_drop(NN->X_train,NN->a_train,NN->h_train, NN->s_l,NN->Theta,v_mom,logistic_unit_prime,Euclidean_CF_prime,size_train_file,batch_size,index,mask,gamma_coef);
		//BP_MBatch_SGD_Adagrad(NN->X_train,NN->a_train,NN->h_train, NN->s_l,NN->Theta,v_mom,logistic_unit_prime,Euclidean_CF_prime,size_train_file,batch_size,index,mask);
		//BP_MBatch_SGD_RMSprop(NN->X_train,NN->a_train,NN->h_train, NN->s_l,NN->Theta,v_mom,relu_prime,Euclidean_CF_prime,size_train_file,batch_size,index,mask,gamma_coef); // 7000 iter
		//BP_MBatch_SGD_Adadelta(NN->X_train,NN->a_train,NN->h_train, NN->s_l,NN->Theta,v_mom,v_mom_2,logistic_unit_prime,Euclidean_CF_prime,size_train_file,batch_size,index,mask,gamma_coef);
		BP_MBatch_SGD_Adam(NN->X_train,NN->a_train,NN->h_train, NN->s_l,NN->Theta,v_mom,v_mom_2,relu_prime,Euclidean_CF_prime,size_train_file,batch_size,index,mask,i+1);
		//

		// print cv error every now and then
		if (i%1000==0) {
			J_train = Euclidean_CF(NN->X_train,NN->h_train[L_layer-1],size_train_file);
			printf("train cost function after %d iterations : %.5e\n\n",i+1,J_train);
			FP_drop(NN->a_cv  ,NN->h_cv,NN->s_l,NN->Theta,relu,c_output,size_cv_file,mask_FP,mask_VU);
			J_cv =Euclidean_CF(NN->X_cv,NN->h_cv[L_layer-1],size_cv_file);
			printf("cv cost function after %d iterations : %.5e\n\n",i+1,J_cv);
			fflush(stdout);
		}
	}

	// FP for all
	FP_drop(NN->a_train,NN->h_train,NN->s_l,NN->Theta,relu,c_output,size_train_file,mask_FP,mask_VU);
	J_train = Euclidean_CF(NN->X_train,NN->h_train[L_layer-1],size_train_file);
	printf("final training error : %.5e\n",J_train);

	FP_drop(NN->a_cv,NN->h_cv,NN->s_l,NN->Theta,relu,c_output,size_cv_file,mask_FP,mask_VU);
	J_cv = Euclidean_CF(NN->X_cv,NN->h_cv[L_layer-1],size_cv_file);
	printf("final cv error : %.5e\n",J_cv);

	FP_drop(NN->a_test,NN->h_test,NN->s_l,NN->Theta,relu,c_output,size_test_file,mask_FP,mask_VU);
	Q_score(NN->X_test,NN->h_test[L_layer-1],size_test_file);
	Q_score_diff(NN->X_test,NN->h_test[L_layer-1],size_test_file);


    ftime(&end);
    diff = (int) (1000.0 * (end.time - start.time)+ (end.millitm - start.millitm));
    printf("\nNeural Network Computation took %u milliseconds\n", diff);
}


