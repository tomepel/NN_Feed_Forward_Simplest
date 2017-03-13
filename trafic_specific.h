/*
 * trafic_specific.h
 *
 *  Created on: 2 mars 2017
 *      Author: tepelbaum
 */



#ifndef TRAFIC_SPECIFIC_H_
#define TRAFIC_SPECIFIC_H_


void  init_X_NN(char * speed_name, char * file_link, double * X_train);
void read_and_aggregate_speed(char **file, char * file_link, double * X_train);
void clean_speeds(double * X_mean5, double * X, double * FFS,int N_row, int N_col);
void init_Xtrain_i_periph(double * X_train_i, double * X_mean5, int N_row);
void init_Xtrain(double  * X, double ** X_train);
void init_Xcv(double  * X, double ** X_cv);
void init_Xtest(double  * X, double ** X_test);

//void init_Xtrain_cv_test(Neural_Network *NN, int n_file);


#endif /* TRAFIC_SPECIFIC_H_ */
