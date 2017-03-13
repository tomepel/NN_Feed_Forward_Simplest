/*
 * nn_spec.c
 *
 *  Created on: 1 mars 2017
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
#include "general_function.h"
#include "trafic_specific.h"


//////////////////// Logistic unit START ////////////////////////////////
// require a double
double logistic_unit(double x) {
  return 1.0/(1+exp(-x));
}
//////////////////// Logistic unit END //////////////////////////////////


//////////////////// Logistic derivative unit START /////////////////////
// require a double
double logistic_unit_prime(double x) {
  return logistic_unit(x)*(1-logistic_unit(x));
}
//////////////////// Logistic derivative unit END ///////////////////////


//////////////////// Standard Sigmoid unit START ////////////////////////
// require a double
double tanh_unit(double x) {
  return (1.0-exp(-2.0*x))/(1.0+exp(-2.0*x));
}
//////////////////// Standard Sigmoid unit END //////////////////////////


//////////////////// Standard Sigmoid der unit START ////////////////////
// require a double
double tanh_unit_prime(double x) {
  return 1-tanh_unit(x)*tanh_unit(x);
}
//////////////////// Standard Sigmoid der unit END //////////////////////


//////////////////// LeCun Sigmoid unit START ///////////////////////////
// require a double
double tanh_lecun(double x) {
  return tanh_m_twothird*(1.0-exp(-fourthird*x))/(1.0+exp(-fourthird*x));
}
//////////////////// LeCun Sigmoid unit END /////////////////////////////


//////////////////// LeCun Sigmoid der unit START ///////////////////////
// require a double
double tanh_lecun_prime(double x) {
  return twothird/tanh_m_twothird*(tanh_m_twothird*tanh_m_twothird-tanh_lecun(x)*tanh_lecun(x));
}
//////////////////// LeCun Sigmoid der unit END /////////////////////////

//////////////////// Relu unit START ////////////////////////////////////
// require a double
double relu(double x) {
  return fmax(x,0);
}
//////////////////// Relu unit END //////////////////////////////////////

//////////////////// Relu unit der START ////////////////////////////////
// require a double
double relu_prime(double x) {
  return  fmin(floor(x)+1,1);
}
//////////////////// Relu unit der END //////////////////////////////////


//////////////////// LeCun Sigmoid unit START ///////////////////////////
// require a vector (usually a_k^(L) )
// require the length of the vector
// require an index k in [1,s_L]
double softmax_o(double * x, int k,int L) {
  return exp(x[k])/sumexp_func(x,L);
}
//////////////////// LeCun Sigmoid unit END /////////////////////////////



//////////////////// Continuous output function START ////////////////////
// require a vector (usually a_k^(L) )
// require the length of the vector
// require an index k in [1,s_L]
double c_output(double x) {
  return x;
}
//////////////////// Continuous output function END //////////////////////



//////////////////// Derivative Euclidean cost function START ////////////
// require a double
double Euclidean_CF_prime(double a, double b) {
  return  a-b;
}
//////////////////// LDerivative Euclidean cost function END /////////////

//////////////////// Euclidean cost function START //////////////////////
// require a set, the output layer and the size of the sample set
double Euclidean_CF(double ** X, double * h, int size_sample) {
	int i,t;
	double J = 0;
	for (i=0;i<s_output;i++) {
		for (t=0;t<size_sample;t++) {
			J += (X[i+s_input][t]- h[i+1+(s_output+1)* t])*(X[i+s_input][t]- h[i+1+(s_output+1)* t]);
		}
	}
	return 0.5*J/(double)size_sample;
}
//////////////////// LDerivative Euclidean cost function END ////////////

// require a set, the output layer and the size of the sample set
void Q_score(double ** X, double * h, int size_sample) {
	int i,j,t;
	double J_prediction;
	double J_benchmark;
	double J_prediction_tot = 0;
	double J_benchmark_tot = 0;
	for (i=0;i<s_output;i++) {
		J_prediction = 0;
		J_benchmark = 0;
		for (t=0;t<size_sample;t++) {
			J_prediction += (X[i+s_input][t]- fmin(1,fmax(0,h[i+1+(s_output+1)* t])))*(X[i+s_input][t]- fmin(1,fmax(0,h[i+1+(s_output+1)* t])));
			J_benchmark  += (X[i+s_input][t]- X[INDEX_ARC][t])*(X[i+s_input][t]-  X[INDEX_ARC][t]);
		}
		J_benchmark_tot += J_benchmark;
		J_prediction_tot += J_prediction;
		printf("Qbench for %d mn prediction : %.4e\n",15*(i+1),1.0-J_prediction/J_benchmark);
	}
	printf("Qbench tot : %.4e\n",1.0-J_prediction_tot/J_benchmark_tot);

	int index;
	int batch_size = 54;
	FILE *stream;
	char filename[100];
	for (j =0;j<batch_size;j++) {
		sprintf(filename, "first_results_%d.csv",j);
		stream= fopen(filename, "w" );
		fprintf(stream,"time;benchmark;prediction;true;\n");
		index=(int) (round(drand48()*size_sample));
		for (i=0;i<s_output;i++) {
			fprintf(stream,"%d;%.4f;%.4f;%.4f;\n",15*(i+1),X[INDEX_ARC][index],fmin(1,fmax(0,h[i+1+(s_output+1)* index])),X[i+s_input][index]);
		}
		fclose(stream);
	}
}
//////////////////// LDerivative Euclidean cost function END ////////////

// require a set, the output layer and the size of the sample set
void Q_score_diff(double ** X, double * h, int size_sample) {
	int i,j,t;
	double J_prediction_FFS;
	double J_prediction_congested;
	double J_prediction_stopped;
	double J_benchmark_FFS;
	double J_benchmark_congested;
	double J_benchmark_stopped;
	double J_prediction_tot = 0;
	double J_benchmark_tot = 0;
	for (i=0;i<s_output;i++) {
		J_prediction_FFS = 0;
		J_prediction_congested = 0;
		J_prediction_stopped = 0;
		J_benchmark_FFS = 0;
		J_benchmark_congested =0;
		J_benchmark_stopped =0;
		for (t=0;t<size_sample;t++) {
			if ( (X[i+s_input][t]>0.5) & (X[i+s_input][t]<=1.0) ) {
				J_prediction_FFS += (X[i+s_input][t]- fmin(1,fmax(0,h[i+1+(s_output+1)* t])))*(X[i+s_input][t]- fmin(1,fmax(0,h[i+1+(s_output+1)* t])));
				J_benchmark_FFS  += (X[i+s_input][t]- X[INDEX_ARC][t])*(X[i+s_input][t]-  X[INDEX_ARC][t]);
			}
			else if (X[i+s_input][t]>0.3) {
				J_prediction_congested += (X[i+s_input][t]- fmin(1,fmax(0,h[i+1+(s_output+1)* t])))*(X[i+s_input][t]- fmin(1,fmax(0,h[i+1+(s_output+1)* t])));
				J_benchmark_congested  += (X[i+s_input][t]- X[INDEX_ARC][t])*(X[i+s_input][t]-  X[INDEX_ARC][t]);
			}
			else {
				J_prediction_stopped += (X[i+s_input][t]- fmin(1,fmax(0,h[i+1+(s_output+1)* t])))*(X[i+s_input][t]- fmin(1,fmax(0,h[i+1+(s_output+1)* t])));
				J_benchmark_stopped  += (X[i+s_input][t]- X[INDEX_ARC][t])*(X[i+s_input][t]-  X[INDEX_ARC][t]);
			}
		}
		J_prediction_tot += J_prediction_FFS+J_prediction_congested+J_prediction_stopped;
		J_benchmark_tot += J_benchmark_FFS+J_benchmark_congested+J_benchmark_stopped;
		printf("Qbench for %d mn prediction : FFS %.4e congested %.4e stopped %.4e\n",15*(i+1),1.0-J_prediction_FFS/J_benchmark_FFS,1.0-J_prediction_congested/J_benchmark_congested,1.0-J_prediction_stopped/J_benchmark_stopped);
	}
	printf("Qbench tot : %.4e\n",1.0-J_prediction_tot/J_benchmark_tot);

	int index;
	int batch_size = 54;
	FILE *stream;
	char filename[100];
	for (j =0;j<batch_size;j++) {
		sprintf(filename, "first_results_%d.csv",j);
		stream= fopen(filename, "w" );
		fprintf(stream,"time;benchmark;prediction;true;\n");
		index=(int) (round(drand48()*size_sample));
		for (i=0;i<s_output;i++) {
			fprintf(stream,"%d;%.4f;%.4f;%.4f;\n",15*(i+1),X[INDEX_ARC][index],fmin(1,fmax(0,h[i+1+(s_output+1)* index])),X[i+s_input][index]);
		}
		fclose(stream);
	}
}
//////////////////// LDerivative Euclidean cost function END ////////////

//////////////////// Euclidean cost function START //////////////////////
// require a set, the output layer and the size of the sample set
double Euclidean_CF_mini_batch(double ** X, double * h, int batch_size,int * index) {
	int i,t,u;
	double J = 0;
	for (i=0;i<s_output;i++) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			J += (X[i+s_input][t]- h[i+1+(s_output+1)* t])*(X[i+s_input][t]- h[i+1+(s_output+1)* t]);
		}
	}
	return 0.5*J/(double)batch_size;
}
//////////////////// LDerivative Euclidean cost function END ////////////


//////////////////// Initialize the weight matrix START /////////////////
// prescription provided by Glorot, Bengio 2010 : sqrt( 6/(s_l+s_lp1) )
// require the Weight matrix (initialization performed layer by layer)
// require the sizes of the tow layers involved
void Init_Theta(double * Theta, int s_l,int s_lp1) {
	int i,j;
	for (i=0;i<s_lp1;i++) {
		for(j=1;j<s_l+1;j++) {
			Theta[j+(s_l+1)*i]=sqrt(6.0/(s_l+s_lp1))*drand48(); //Weights
		}
		Theta[0+(s_l+1)*i]=0; // biases
	}
}
//////////////////// Initialize the weight matrix END ///////////////////



//////////////////// Initialize the momentum matrix START ///////////////
// prescription provided by Glorot, Bengio 2010 : sqrt( 6/(s_l+s_lp1) )
// require the Weight matrix (initialization performed layer by layer)
// require the sizes of the tow layers involved
void Init_v_mom(double * v_mom, int s_l,int s_lp1) {
	int i,j;
	for (i=0;i<s_lp1;i++) {
		for(j=0;j<s_l+1;j++) {
			v_mom[j+(s_l+1)*i]=0;
		}
	}
}
//////////////////// Initialize the momentum matrix END /////////////////




/////////////////////////// INIT NN START ///////////////////////////////
// Return initialized NN
// Need number of speed files
// Need speed filenames
// Need link filename
// require a (0,1) int (0 = no header, 1 = header)
void Init_NN(Neural_Network *NN, char * speed_name, char * link_name) {
	// miscellaneous variables
	int j,k,l;

	// Initialize the input variables (tidy but not yet in sets)
	NN->X = (double *)malloc(N_FILE*size_tot_X*sizeof(double));
	init_X_NN(speed_name,link_name,NN->X);

	// Initialize the different layer's dimensions
	NN->s_l = (int*)malloc((L_layer)*sizeof(int));
	NN->s_l[0] = s_input;
	printf("size of input layer %d initialized at %d \n",0,	NN->s_l[0]);
	for (l=1;l<L_layer-1;l++) {
		NN->s_l[l]=s_hidden;
		printf("size of hidden layer %d initialized at %d \n",l,	NN->s_l[l]);
	}
	NN->s_l[L_layer-1] = s_output;
	printf("size of output layer %d initialized at %d \n",L_layer-1,	NN->s_l[L_layer-1]);
	printf("\n");

	// Initialize the weight matrix
	NN->Theta = (double **)malloc((L_layer-1)*sizeof (double));
	for (l= 0;l<L_layer-1;l++) {
		NN->Theta[l] = (double *)malloc((NN->s_l[l+1])*(NN->s_l[l]+1)*sizeof (double));
		Init_Theta(NN->Theta[l],NN->s_l[l],NN->s_l[l+1]);
		printf("Weight between layer %d and %d initialized, matrix of dimension %d X %d \n",l,l+1,NN->s_l[l+1],NN->s_l[l]+1);
	}
	printf("\n");

	// Initialize the a's (see note on NN)
	NN->a_train = (double **)malloc((L_layer-1)*sizeof (double));
	NN->a_cv = (double **)malloc((L_layer-1)*sizeof (double));
	NN->a_test = (double **)malloc((L_layer-1)*sizeof (double));
	for (l= 0;l<L_layer-1;l++) {
		NN->a_train[l] = (double *)malloc(	(NN->s_l[l+1])*size_train_file*sizeof (double));
		NN->a_cv[l] = (double *)malloc(		(NN->s_l[l+1])*size_cv_file*sizeof (double));
		NN->a_test[l] = (double *)malloc(	(NN->s_l[l+1])*size_test_file*sizeof (double));
	}
	printf("Linear combinations a correctly initialized \n\n");

	// Initialize the h's (see note on NN)
	NN->h_train = (double **)malloc((L_layer)*sizeof (double));
	NN->h_cv = (double **)malloc((L_layer)*sizeof (double));
	NN->h_test = (double **)malloc((L_layer)*sizeof (double));
	for (l= 0;l<L_layer;l++) {
		NN->h_train[l] = (double *)malloc(	(NN->s_l[l]+1)*size_train_file*sizeof (double));
		NN->h_cv[l] = (double *)malloc(		(NN->s_l[l]+1)*size_cv_file*sizeof (double));
		NN->h_test[l] = (double *)malloc(	(NN->s_l[l]+1)*size_test_file*sizeof (double));
	}

	printf("activation variables correctly initialized \n\n");

	// Initialize the X_train
	NN->X_train = (double **)malloc((s_input+s_output)*sizeof(double));
	for (j=0;j<s_input+s_output;j++) {
		NN->X_train[j] = (double *)malloc(size_train_file*sizeof (double));
	}
	init_Xtrain(NN->X,NN->X_train);
	printf("X_train correctly affected \n\n");

	for(j = 0;j<size_train_file;j++) {
		//NN->h_train[0][0+(NN->s_l[0]+1)* j] = 1;
		for(k = 1;k<NN->s_l[0]+1;k++) {
			NN->h_train[0][k+(NN->s_l[0]+1)* j] =NN->X_train[k-1][j];
		}
	}

	for (l=0;l<L_layer;l++) { // loop on layer l in 0, L-1
		for (j=0;j<size_train_file;j++) { // loop on sample size (change for training, cv, and test)
			NN->h_train[l][0+(NN->s_l[l]+1)*j] = 1.0;
		}
	}

	// Initialize the X_cv
	NN->X_cv = (double **)malloc((s_input+s_output)*sizeof(double));
	for (j=0;j<s_input+s_output;j++) {
		NN->X_cv[j] = (double *)malloc(size_cv_file*sizeof (double));
	}
	init_Xcv(NN->X,NN->X_cv);
	printf("X_cv correctly affected \n\n");

	for(j = 0;j<size_cv_file;j++) {
		//NN->h_cv[0][0+(NN->s_l[0]+1)* j] = 1;
		for(k = 1;k<NN->s_l[0]+1;k++) {
			NN->h_cv[0][k+(NN->s_l[0]+1)* j] =NN->X_cv[k-1][j];
		}
	}

	for (l=0;l<L_layer;l++) { // loop on layer l in 0, L-1
		for (j=0;j<size_cv_file;j++) { // loop on sample size (change for training, cv, and test)
			NN->h_cv[l][0+(NN->s_l[l]+1)*j] = 1.0;
		}
	}

	// Initialize the X_test
	NN->X_test = (double **)malloc((s_input+s_output)*sizeof(double));
	for (j=0;j<s_input+s_output;j++) {
		NN->X_test[j] = (double *)malloc(size_test_file*sizeof (double));
	}
	init_Xtest(NN->X,NN->X_test);
	printf("X_test correctly affected \n\n");

	for(j = 0;j<size_test_file;j++) {
		//NN->h_test[0][0+(NN->s_l[0]+1)* j] = 1;
		for(k = 1;k<NN->s_l[0]+1;k++) {
			NN->h_test[0][k+(NN->s_l[0]+1)* j] =NN->X_test[k-1][j];
		}
	}

	for (l=0;l<L_layer;l++) { // loop on layer l in 0, L-1
		for (j=0;j<size_test_file;j++) { // loop on sample size (change for training, cv, and test)
			NN->h_test[l][0+(NN->s_l[l]+1)*j] = 1.0;
		}
	}

	// Check if file has been correctly loaded
	/*printf("\n");
	for (k=0;k<20;k++) {
		for (j=0;j<s_input+s_output;j++) {
			printf("%.13e ",NN->X_cv[j][k]);
		}
		printf("\n");
	}
	printf("\n");
	 for (k=size_file-10;k<size_file;k++) {
		for (j=0;j<s_input+s_output;j++) {
			printf("%.13e ",NN->X[j+(s_input+s_output)*k+9*size_file*(s_input+s_output)]);
		}
		printf("\n");
	}*/
}
/////////////////////////// INIT NN END /////////////////////////////////



/////////////////////////// Forward pass START //////////////////////////
void Forward_Propagation(double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double), double (*output_unit)(double), int size_sample) {
	int i,j,t,l;
	for (l=0;l<L_layer-2;l++) { // loop on layer l in 0, L-1
		for (t=0;t<size_sample;t++) { // loop on sample size (change for training, cv, and test)
			// initialize the activation as
			for (i=0;i<sl[l+1];i++) { // loop on size of layer l+1
				a[l][i+(sl[l+1])*t] = 0;
				for (j=0;j<sl[l]+1;j++) {
					a[l][i+(sl[l+1])*t] += Theta[l][j+(sl[l]+1)*i]*h[l][j+(sl[l]+1)*t];
				}
			}

			// initialize the hidden layers
			for (j=1;j<sl[l+1]+1;j++) {
				h[l+1][j+(sl[l+1]+1)*t] =hidden_unit(a[l][j-1+(sl[l+1])*t]);
			}
		}
	}

	l = L_layer-2;
	for (t=0;t<size_sample;t++) { // loop on sample size (change for training, cv, and test)
		// initialize the activation as
		for (i=0;i<sl[l+1];i++) { // loop on size of layer l+1
			a[l][i+(sl[l+1])*t] = 0;
			for (j=0;j<sl[l]+1;j++) {
				a[l][i+(sl[l+1])*t] += Theta[l][j+(sl[l]+1)*i]*h[l][j+(sl[l]+1)*t];
			}
		}

		// initialize the hidden layers
		for (j=1;j<sl[l+1]+1;j++) {
			h[l+1][j+(sl[l+1]+1)*t] =output_unit(a[l][j-1+(sl[l+1])*t]);
		}
	}
}
/////////////////////////// Forward pass END ////////////////////////////



/////////////////////////// Forward pass START //////////////////////////
void FP_drop(double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double), double (*output_unit)(double), int size_sample, double mask, double mask_VU) {
	int i,j,t,l;
	//#pragma omp for
	for (t=0;t<size_sample;t++) {
		l=0;
		for (i=0;i<sl[l+1];i++) {
			a[l][i+(sl[l+1])*t] = 0;
			for (j=0;j<sl[l]+1;j++) {
				a[l][i+(sl[l+1])*t] += mask_VU*Theta[l][j+(sl[l]+1)*i]*h[l][j+(sl[l]+1)*t];
			}
		}
		for (j=1;j<sl[l+1]+1;j++) {
			h[l+1][j+(sl[l+1]+1)*t] = hidden_unit(a[l][j-1+(sl[l+1])*t]);
		}

		for (l=1;l<L_layer-2;l++) {
			for (i=0;i<sl[l+1];i++) {
				a[l][i+(sl[l+1])*t] = Theta[l][(sl[l]+1)*i]*h[l][(sl[l]+1)*t];
				for (j=1;j<sl[l]+1;j++) {
					a[l][i+(sl[l+1])*t] += mask*Theta[l][j+(sl[l]+1)*i]*h[l][j+(sl[l]+1)*t];
				}
			}
			for (j=1;j<sl[l+1]+1;j++) {
				h[l+1][j+(sl[l+1]+1)*t] = hidden_unit(a[l][j-1+(sl[l+1])*t]);
			}
		}

		l = L_layer-2;
		for (i=0;i<sl[l+1];i++) {
			a[l][i+(sl[l+1])*t] = Theta[l][(sl[l]+1)*i]*h[l][(sl[l]+1)*t];
			for (j=1;j<sl[l]+1;j++) {
				a[l][i+(sl[l+1])*t] += mask*Theta[l][j+(sl[l]+1)*i]*h[l][j+(sl[l]+1)*t];
			}
		}
		for (j=1;j<sl[l+1]+1;j++) {
			h[l+1][j+(sl[l+1]+1)*t] =output_unit(a[l][j-1+(sl[l+1])*t]);
		}

	}
}
/////////////////////////// Forward pass END ////////////////////////////


/////////////////////////// Forward pass mini_batch START ///////////////
void Forward_Propagation_mini_batch(double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),
		double (*output_unit)(double),int batch_size, int * index) {
	int i,j,t,u,l;
	//printf("index in forward_prop %d\n",index[0]);
	for (l=0;l<L_layer-1;l++) { // loop on layer l in 0, L-1
		for (u=0;u<batch_size;u++) { // loop on sample size (change for training, cv, and test)
			t = index[u];
			// initialize the biases of the different layers
			//h[l+1][0+(sl[l+1]+1)*t] = 1.0;

			// initialize the activation as
			for (i=0;i<sl[l+1];i++) { // loop on size of layer l+1
				a[l][i+(sl[l+1])*t] = 0;
				for (j=0;j<sl[l]+1;j++) {
					a[l][i+(sl[l+1])*t] += Theta[l][j+(sl[l]+1)*i]*h[l][j+(sl[l]+1)*t];
				}
			}

			// initialize the hidden layers
			if (l!=L_layer-2) {
				for (j=1;j<sl[l+1]+1;j++) {
					h[l+1][j+(sl[l+1]+1)*t] =hidden_unit(a[l][j-1+(sl[l+1])*t]);
				}
			}
			else {
				for (j=1;j<sl[l+1]+1;j++) {
					h[l+1][j+(sl[l+1]+1)*t] =output_unit(a[l][j-1+(sl[l+1])*t]);
				}
			}
		}
	}
}
/////////////////////////// Forward pass mini_batch END /////////////////


/////////////////////////// Forward pass mini_batch START ///////////////
void FP_MBatch_drop(double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),double (*output_unit)(double),int batch_size, int * index, int ** mask) {
	int i,j,t,u,l;
	for (u=0;u<batch_size;u++) {
		l=0;
		t = index[u];
		for (j=1;j<sl[l+1]+1;j++) {
			h[l][j+(sl[l]+1)*t] =mask[l][j-1+(sl[l])*u]*h[l][j+(sl[l]+1)*t];
		}

		for (l=0;l<L_layer-2;l++) {
			for (i=0;i<sl[l+1];i++) {
				a[l][i+(sl[l+1])*t] = 0;
				for (j=0;j<sl[l]+1;j++) {
					a[l][i+(sl[l+1])*t] += Theta[l][j+(sl[l]+1)*i]*h[l][j+(sl[l]+1)*t];
				}
			}
			for (j=1;j<sl[l+1]+1;j++) {
				h[l+1][j+(sl[l+1]+1)*t] =mask[l+1][j-1+(sl[l+1])*u]*hidden_unit(a[l][j-1+(sl[l+1])*t]);
			}
		}
		l=L_layer-2;
		for (i=0;i<sl[l+1];i++) {
			a[l][i+(sl[l+1])*t] = 0;
			for (j=0;j<sl[l]+1;j++) {
				a[l][i+(sl[l+1])*t] += Theta[l][j+(sl[l]+1)*i]*h[l][j+(sl[l]+1)*t];
			}
		}
		for (j=1;j<sl[l+1]+1;j++) {
			h[l+1][j+(sl[l+1]+1)*t] =output_unit(a[l][j-1+(sl[l+1])*t]);
		}
	}

}
/////////////////////////// Forward pass mini_batch END /////////////////


/////////////////////////// Standard Batch Backprop START ///////////////
// Standard backprop algorithm
void Backpropagation_Batch_SGD(double **X, double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),double (*output_unit)(double, double), int size_sample) {
	int i,j,t,l;
	int kronecker;
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));
	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*size_sample*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (t=0;t<size_sample;t++) {
		//delta[L_layer-2][0+(sl[L_layer-1]+1)*t] =output_unit(h[L_layer-1][0+(sl[L_layer-1]+1)*t],0);
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*t] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (t=0;t<size_sample;t++) {
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*t] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i]
					delta[l-1][i+sl[l]*t] +=Theta[l][i+1+(sl[l]+1)*j]*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*t];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)size_sample)*Theta[l][j+(sl[l]+1)*i];
				for (t=0;t<size_sample;t++) {
					Theta[l][j+(sl[l]+1)*i] -=ETA*(h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*t])/(double)size_sample;
				}
			}
		}
	}


	free(delta);
}
/////////////////////////// Standard Batch Backprop END /////////////////


/////////////////////////// Mini Batch Backprop START ///////////////////
// Mini Batch backprop algorithm
void Backpropagation_Mini_Batch_SGD(double **X, double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index) {
	int i,j,t,l,u;
	int kronecker;
	//printf("index in backprop %d\n",index[0]);
	/*int index[batch_size];
	for (i =0;i<batch_size;i++) {
		index[i]=(int) (round(drand48()*size_sample));
	}*/
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));
	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		//delta[L_layer-2][0+(sl[L_layer-1]+1)*t] =output_unit(h[L_layer-1][0+(sl[L_layer-1]+1)*t],0);
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji
					delta[l-1][i+sl[l]*u] +=Theta[l][i+1+(sl[l]+1)*j]*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				for (u=0;u<batch_size;u++) {
					t = index[u];
					Theta[l][j+(sl[l]+1)*i] -=ETA*(h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
			}
		}
	}


	free(delta);
}
/////////////////////////// Mini Batch Backprop END /////////////////////



/////////////////////////// Mini Batch Backprop START ///////////////////
// Mini Batch backprop algorithm
void Backpropagation_Mini_Batch_SGD_drop(double **X, double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask) {
	int i,j,t,l,u;
	int kronecker;
	//printf("index in backprop %d\n",index[0]);
	/*int index[batch_size];
	for (i =0;i<batch_size;i++) {
		index[i]=(int) (round(drand48()*size_sample));
	}*/
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));
	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		//delta[L_layer-2][0+(sl[L_layer-1]+1)*t] =output_unit(h[L_layer-1][0+(sl[L_layer-1]+1)*t],0);
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji
					delta[l-1][i+sl[l]*u] +=Theta[l][i+1+(sl[l]+1)*j]*hidden_unit(a[l-1][i+(sl[l])*t])*mask[l][i+(sl[l])*u]*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				for (u=0;u<batch_size;u++) {
					t = index[u];
					Theta[l][j+(sl[l]+1)*i] -=ETA*(h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
			}
		}
	}


	free(delta);
}
/////////////////////////// Mini Batch Backprop END /////////////////////





//////////////// Momentum mini batch Batch Backprop START ///////////////
// Momentum mini batch backprop algorithm
void Backpropagation_Mini_Batch_SGD_momentum(double **X, double **a,double **h, int  * sl, double ** Theta, double ** v_mom, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index) {
	int i,j,t,l,u;
	int kronecker;
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));
	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji
					delta[l-1][i+sl[l]*u] +=Theta[l][i+1+(sl[l]+1)*j]*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				v_mom[l][j+(sl[l]+1)*i] = GAMMA*v_mom[l][j+(sl[l]+1)*i];
				for (u=0;u<batch_size;u++) {
					t = index[u];
					v_mom[l][j+(sl[l]+1)*i] += ETA*(h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
				Theta[l][j+(sl[l]+1)*i] -=v_mom[l][j+(sl[l]+1)*i];
			}
		}
	}


	free(delta);
}
//////////////// Momentum mini batch Batch Backprop END /////////////////




//////////////// NAG mini batch Batch Backprop START ////////////////////
// NAG mini batch backprop algorithm
void Backpropagation_Mini_Batch_SGD_NAG(double **X, double **a,double **h, int  * sl, double ** Theta, double ** v_mom, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index) {
	int i,j,t,l,u;
	int kronecker;
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));

	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji, prescience step of NAG
					delta[l-1][i+sl[l]*u] +=(Theta[l][i+1+(sl[l]+1)*j]-GAMMA*v_mom[l][j+(sl[l]+1)*i])*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				v_mom[l][j+(sl[l]+1)*i] = GAMMA*v_mom[l][j+(sl[l]+1)*i];
				for (u=0;u<batch_size;u++) {
					t = index[u];
					v_mom[l][j+(sl[l]+1)*i] += ETA*(h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
				Theta[l][j+(sl[l]+1)*i] -=v_mom[l][j+(sl[l]+1)*i];
			}
		}
	}


	free(delta);
}
//////////////// NAG mini batch Batch Backprop END //////////////////////




//////////////// NAG mini batch Batch Backprop drop START ///////////////
// NAG mini batch backprop algorithm
void BP_MBatch_SGD_NAG_drop(double **X, double **a,double **h, int  * sl, double ** Theta, double ** v_mom, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask, int gamma_coef) {
	int i,j,t,l,u;
	int kronecker;
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));

	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji, prescience step of NAG
					delta[l-1][i+sl[l]*u] +=(Theta[l][i+1+(sl[l]+1)*j]-gamma_coef*v_mom[l][j+(sl[l]+1)*i])*mask[l][i+(sl[l])*u]*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				v_mom[l][j+(sl[l]+1)*i] = gamma_coef*v_mom[l][j+(sl[l]+1)*i];
				for (u=0;u<batch_size;u++) {
					t = index[u];
					v_mom[l][j+(sl[l]+1)*i] += ETA*(h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
				Theta[l][j+(sl[l]+1)*i] -=v_mom[l][j+(sl[l]+1)*i];
			}
		}
	}


	free(delta);
}
//////////////// Momentum mini batch Batch Backprop END //////////////////


//////////////// Adagrad mini batch Batch Backprop drop START ///////////////
// Adagrad mini batch backprop algorithm
void BP_MBatch_SGD_Adagrad(double **X, double **a,double **h, int  * sl, double ** Theta, double ** v_mom, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask) {
	int i,j,t,l,u;
	int kronecker;
	double g_tij;
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));

	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji, prescience step of NAG
					delta[l-1][i+sl[l]*u] +=(Theta[l][i+1+(sl[l]+1)*j])*mask[l][i+(sl[l])*u]*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				g_tij = 0;
				for (u=0;u<batch_size;u++) {
					t = index[u];
					g_tij +=  (h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
				v_mom[l][j+(sl[l]+1)*i] += g_tij*g_tij;
				Theta[l][j+(sl[l]+1)*i] -=ETA*g_tij/sqrt(v_mom[l][j+(sl[l]+1)*i]+EPSILON);
			}
		}
	}


	free(delta);
}
//////////////// Momentum mini batch Batch Backprop END //////////////////



//////////////// RMSPROP mini batch Batch Backprop drop START ///////////////
// RMSPROP mini batch backprop algorithm
void BP_MBatch_SGD_RMSprop(double **X, double **a,double **h, int  * sl, double ** Theta, double ** RMS_g, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask, int gamma_coef) {
	int i,j,t,l,u;
	int kronecker;
	double g_tij;
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));

	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji, prescience step of NAG
					delta[l-1][i+sl[l]*u] +=(Theta[l][i+1+(sl[l]+1)*j])*mask[l][i+(sl[l])*u]*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				g_tij = 0;
				for (u=0;u<batch_size;u++) {
					t = index[u];
					g_tij +=  (h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
				RMS_g[l][j+(sl[l]+1)*i] = gamma_coef*RMS_g[l][j+(sl[l]+1)*i]+(1-gamma_coef)*g_tij*g_tij;
				Theta[l][j+(sl[l]+1)*i] -=ETA*g_tij/sqrt(RMS_g[l][j+(sl[l]+1)*i]+EPSILON);
			}
		}
	}


	free(delta);
}
//////////////// Momentum mini batch Batch Backprop END //////////////////



//////////////// Adadelta mini batch Batch Backprop drop START ///////////////
// Adadelta mini batch backprop algorithm
void BP_MBatch_SGD_Adadelta(double **X, double **a,double **h, int  * sl, double ** Theta, double ** RMS_g, double ** RMS_theta, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask, int gamma_coef) {
	int i,j,t,l,u;
	int kronecker;
	double g_tij,delta_theta_ij;
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));

	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji, prescience step of NAG
					delta[l-1][i+sl[l]*u] +=(Theta[l][i+1+(sl[l]+1)*j])*mask[l][i+(sl[l])*u]*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				g_tij = 0;
				for (u=0;u<batch_size;u++) {
					t = index[u];
					g_tij +=  (h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
				RMS_g[l][j+(sl[l]+1)*i] =gamma_coef*RMS_g[l][j+(sl[l]+1)*i]+ (1-gamma_coef)*g_tij*g_tij;
				delta_theta_ij=sqrt(RMS_theta[l][j+(sl[l]+1)*i]+EPSILON)*g_tij/sqrt(RMS_g[l][j+(sl[l]+1)*i]+EPSILON);
				RMS_theta[l][j+(sl[l]+1)*i]  =gamma_coef*RMS_theta[l][j+(sl[l]+1)*i]+ (1-gamma_coef)*delta_theta_ij*delta_theta_ij;
				Theta[l][j+(sl[l]+1)*i] -=delta_theta_ij;

			}
		}
	}


	free(delta);
}
//////////////// Momentum mini batch Batch Backprop END //////////////////



//////////////// Adam mini batch Batch Backprop drop START ///////////////
// Adam mini batch backprop algorithm
void BP_MBatch_SGD_Adam(double **X, double **a,double **h, int  * sl, double ** Theta, double ** mt, double ** vt, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask, int idx) {
	int i,j,t,l,u;
	int kronecker;
	double g_tij;
	// delta (l in [0,L-2]) here corresponds to delta (l+2 in [2,L]) in my note
	double ** delta =(double **)malloc((L_layer-1)*sizeof(double));

	for (l=L_layer-2;l>=0;l--) {
		delta[l] = (double *)malloc((sl[l+1])*batch_size*sizeof (double));
	}
	// need to initialize delta[L-2] (delta at the last layer), depends on the cost function !!!
	for (u=0;u<batch_size;u++) {
		t = index[u];
		for (i=0;i<sl[L_layer-1];i++) {
			delta[L_layer-2][i+sl[L_layer-1]*u] =output_unit(h[L_layer-1][i+1+(sl[L_layer-1]+1)*t],X[s_input+i][t]);// bias left out
		}
	}

	// needs the derivative of the hidden layer function
	for (l=L_layer-2;l>0;l--) {
		for (u=0;u<batch_size;u++) {
			t = index[u];
			for (i=0;i<sl[l];i++) {
				delta[l-1][i+sl[l]*u] =0;
				for (j=1;j<sl[l+1]+1;j++) { // bias left out
					// Recall that Theta_ij is Theta[j+(s_l+1)*i], thus subtlety Theta_ji, prescience step of NAG
					delta[l-1][i+sl[l]*u] +=(Theta[l][i+1+(sl[l]+1)*j])*mask[l][i+(sl[l])*u]*hidden_unit(a[l-1][i+(sl[l])*t])*delta[l][j-1+(sl[l+1])*u];
				}
			}
		}
	}

	for (l=0;l<L_layer-1;l++) {
		for (i=0;i<sl[l+1];i++) {
			for (j=0;j<sl[l]+1;j++) {
				kronecker = (j==0) ? 0 : 1;
				Theta[l][j+(sl[l]+1)*i] =(1.0-kronecker*ETA*LAMBDA/(double)batch_size)*Theta[l][j+(sl[l]+1)*i];
				g_tij = 0;
				for (u=0;u<batch_size;u++) {
					t = index[u];
					g_tij +=  (h[l][j+(sl[l]+1)*t]*delta[l][i+(sl[l+1])*u])/(double)batch_size;
				}
				mt[l][j+(sl[l]+1)*i] =BETA_1*mt[l][j+(sl[l]+1)*i]+ (1-BETA_1)*g_tij;
				vt[l][j+(sl[l]+1)*i] =BETA_2*vt[l][j+(sl[l]+1)*i]+ (1-BETA_2)*g_tij*g_tij;
				Theta[l][j+(sl[l]+1)*i] -=ETA*mt[l][j+(sl[l]+1)*i]/( ( sqrt(vt[l][j+(sl[l]+1)*i]/(1-pow(BETA_2,idx)))+EPSILON)*(1-pow(BETA_1,idx)) );

			}
		}
	}


	free(delta);
}
//////////////// Momentum mini batch Batch Backprop END //////////////////
