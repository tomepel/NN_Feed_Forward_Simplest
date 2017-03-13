/*
 * trafic_specific.c
 *
 *  Created on: 2 mars 2017
 *      Author: tepelbaum
 */


#include <math.h>
#include <omp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
/*#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>*/
#include <fcntl.h>



#include "macro.h"
#include "structs.h"
#include "file_treatment.h"
#include "general_function.h"
#include "trafic_specific.h"

///////////////////////////////////////////// Init the X NN START /////////////////////////////////////////////////////////////
// Needs and returns Return X_NN
// Need speed filenames
// Need link filename
// require a (0,1) int (0 = no header, 1 = header)
// Need number of speed files
void init_X_NN(char * speed_name, char * file_link, double * X_train) {

	// Misceallaneous variables
	int i;
	//double ** X_NN = (double **)malloc(N_FILE*sizeof(double));
	// intialize the different file names
	char **file= (char **)malloc(N_FILE*sizeof(char));
	char beg[25];
	char mid[25];
	int n_init = S_DAY;
	char end []= ".csv";
	for (i=n_init;i<n_init+N_FILE;i++) {
		strcpy(beg,speed_name);
		sprintf(mid, "%d", i);
		strcat(beg ,mid);
		strcat(beg ,end);
		file[i-n_init] = (char *)malloc(100*sizeof(char));
		strcpy(file[i-n_init],beg);
	}
	// initialize the tidy X element of the NN
	read_and_aggregate_speed(file,file_link, X_train);
	printf("%d Files correctly loaded, X_NN initialized\n\n",N_FILE);
	//return X_NN;
}
///////////////////////////////////////////// Init the X NN END ///////////////////////////////////////////////////////////////



///////////////////////////////////////////// Load and clean the speed data START /////////////////////////////////////////////
// need a list of speed files
// need a file of links
// need the number of speed files
// require a (0,1) int (0 = no header, 1 = header)
// need the X element from the Neural_Network (tidy data, not yet attributed to training, cv and test sets)
void read_and_aggregate_speed(char **file, char * file_link, double * X_train) {

	// miscellaneous variables
	int i;
	int N_col, N_row;
	/* X;
	double * X_mean5;*/

	// load the link_file, extract the FFS
	N_col = csv_num_column(file_link);
	N_row  = csv_num_line(file_link);
	double  * link = (double *)malloc(N_col*N_row*sizeof (double));
	read_csv(file_link,link, N_col);

	// define the FFS, assume that it is the last column of the link_file
	double  * FFS = (double *)malloc(N_row*sizeof (double));
	for (i=0;i<N_row;i++) {
		FFS[i] =link[N_col*i+N_col-1];
	}

	// for each file, extract and store relevant information (normalized speed)
	for (i=0;i<N_FILE;i++) {
		N_row  = csv_num_line(file[i]);
		N_col = csv_num_column(file[i]);
		double  *X =  (double *)malloc(N_col*N_row*sizeof (double));
		double  * X_mean5 = (double *)malloc(N_1_over_4_h*N_row*sizeof (double));
		read_csv(file[i],X, N_col);

		clean_speeds(X_mean5,X,FFS,N_row,N_col);
		//X_train[i] = (double *)malloc((N_row*T_size_training_dat)*(s_input+s_output)*sizeof (double));
		init_Xtrain_i_periph(X_train+i*size_file*(s_input+s_output),X_mean5,N_row);
		free(X);
		free(X_mean5);
	}

	// clean
	//return X_train;
	free(FFS);
	free(link);
}
///////////////////////////////////////////// Load and clean the speed data END ///////////////////////////////////////////////



///////////////////////////////////////////// mean by quarter hour START //////////////////////////////////////////////////////
// Need a list to store the means
// Need the data from a file
// Need the FFS
// Need the number of rows of the file
// Nee the number of colmuns of the file
void clean_speeds(double * X_mean5, double * X, double * FFS,int N_row, int N_col) {

	// miscellaneous variables
	int i,j;

	// normalize speed
	for (i=0;i<N_row;i++) {
		for (j=0;j<N_col;j++) {
			X[j+N_col*i] = X[j+N_col*i]/FFS[i];
		}
	}

	// mean speed from five 3 mn intervals to one 15 mn interval
	for (i=0;i<N_row;i++) {
		for (j=0;j<N_1_over_4_h;j++) {
			X_mean5[j+N_1_over_4_h*i] = mean_func(X+N_col*i+start_index+5*j,5);
			if (X_mean5[j+N_1_over_4_h*i] < 0.01) { // BEWARE OF 0 DATA !!
				X_mean5[j+N_1_over_4_h*i] = 1;
			}
		}
	}
}
///////////////////////////////////////////// mean by quarter hour END ////////////////////////////////////////////////////////



///////////////////////////////////////////// Three  Neighbours affect X of NN START //////////////////////////////////////////
// Return the ith element of the X element of NN (tidy data, not yet attributed to training, cv and test sets)
// Need the mean by quarter hour for the ith file
// Need the number of rows of the ith file
void init_Xtrain_i_periph(double * X_train_i, double * X_mean5, int N_row) {

	// miscellaneous variables
	int i,t;
	int s_tr = s_input/N_Neighbours;
	int avant, apres;

	// Affects to the input variables last s_tr known time for the N_Neighbours (including itself) surrounding arcs
	for(i=0;i<N_row;i++) {
		avant = i-1;
		apres = i+1;
		if (i==0) avant = N_row -1;
		if (i==N_row-1) apres  = 1;
		for (t=0;t<T_size_training_dat;t++) {
			memcpy(X_train_i+(s_input+s_output)*(t+T_size_training_dat*i)					,X_mean5+avant*N_1_over_4_h+t 	,s_tr*sizeof(double));
			memcpy(X_train_i+(s_input+s_output)*(t+T_size_training_dat*i)	+s_tr			,X_mean5+i*N_1_over_4_h+t 		,s_tr*sizeof(double));
			memcpy(X_train_i+(s_input+s_output)*(t+T_size_training_dat*i)	+2*s_tr			,X_mean5+apres*N_1_over_4_h+t 	,s_tr*sizeof(double));
			memcpy(X_train_i+(s_input+s_output)*(t+T_size_training_dat*i)	+s_input		,X_mean5+i*N_1_over_4_h+t+s_tr 	,s_output*sizeof(double));
		}
	}
}
///////////////////////////////////////////// Three  Neighbours affect X of NN END ////////////////////////////////////////////


///////////////////////////////////////////// Initialize X_train START ////////////////////////////////////////////////////////
// Needs tidy data
// Return the initialized X_train element of NN
// Need the number of files
void init_Xtrain(double  * X, double ** X_train) {

	// miscellaneous variables
	int i,j,l;

	for (j=0;j<s_input+s_output;j++) {
		for (l = 0;l<size_training;l++) {
			for (i=0;i<size_file;i++) {
				X_train[j][i+size_file*l] = X[j+(s_input+s_output)*i+l*size_tot_X];
			}
		}
	}
}
///////////////////////////////////////////// Initialize X_train END //////////////////////////////////////////////////////////



///////////////////////////////////////////// Initialize X_cv START ///////////////////////////////////////////////////////////
// Needs tidy data
// Return the initialized X_cv element of NN
// Need the number of files
void init_Xcv(double  * X, double ** X_cv) {

	// miscellaneous variables
	int i,j,l;

	for (j=0;j<s_input+s_output;j++) {
		for (l = 0;l<size_cv;l++) {
			for (i=0;i<size_file;i++) {
				X_cv[j][i+size_file*l] = X[j+(s_input+s_output)*i+(size_training+l)*size_tot_X];
			}
		}
	}
}
///////////////////////////////////////////// Initialize X_cv END /////////////////////////////////////////////////////////////



///////////////////////////////////////////// Initialize X_test START/ ////////////////////////////////////////////////////////
// Needs tidy data
// Return the initialized X_test element of NN
// Need the number of files
void init_Xtest(double  * X, double ** X_test) {

	// miscellaneous variables
	int i,j,l;

	for (j=0;j<s_input+s_output;j++) {
		for (l = 0;l<size_test;l++) {
			for (i=0;i<size_file;i++) {
				X_test[j][i+size_file*l] = X[j+(s_input+s_output)*i+(size_training+size_cv+l)*size_tot_X];
			}
		}
	}
}
///////////////////////////////////////////// Initialize X_test END ///////////////////////////////////////////////////////////
