/*
 * nn_spec.h
 *
 *  Created on: 1 mars 2017
 *      Author: tepelbaum
 */



#ifndef NN_SPEC_H_
#define NN_SPEC_H_



double logistic_unit(double x);
double logistic_unit_prime(double x);
double tanh_unit(double x);
double tanh_unit_prime(double x);
double tanh_lecun(double x);
double tanh_lecun_prime(double x);
double relu(double x);
double relu_prime(double x);
double softmax_o(double * x, int k,int L);
double c_output(double x);
double Euclidean_CF_prime(double a, double b);
double Euclidean_CF(double ** X, double * h, int size_sample);
void Q_score(double ** X, double * h, int size_sample);
void Q_score_diff(double ** X, double * h, int size_sample);
double Euclidean_CF_mini_batch(double ** X, double * h, int batch_size,int * index);
void Init_Theta(double * Theta, int s_l,int s_lp1);
void Init_v_mom(double * v_mom, int s_l,int s_lp1);
void Init_NN(Neural_Network *NN,  char * speed_name, char * link_name);
void Forward_Propagation(double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double), double (*output_unit)(double), int size_sample);
void FP_drop(double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double), double (*output_unit)(double), int size_sample, double mask, double mask_VU);
void Forward_Propagation_mini_batch(double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),
		double (*output_unit)(double),int batch_size, int * index);
void FP_MBatch_drop(double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),
		double (*output_unit)(double),int batch_size, int * index, int ** mask);
void Backpropagation_Batch_SGD(double **X, double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),double (*output_unit)(double, double), int size_sample);
void Backpropagation_Mini_Batch_SGD(double **X, double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index);
void Backpropagation_Mini_Batch_SGD_drop(double **X, double **a,double **h, int  * sl, double ** Theta, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask);
void Backpropagation_Mini_Batch_SGD_momentum(double **X, double **a,double **h, int  * sl, double ** Theta, double ** v_mom, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index);
void Backpropagation_Mini_Batch_SGD_NAG(double **X, double **a,double **h, int  * sl, double ** Theta, double ** v_mom, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index);
void BP_MBatch_SGD_NAG_drop(double **X, double **a,double **h, int  * sl, double ** Theta, double ** v_mom, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask, int gamma_coef) ;
void BP_MBatch_SGD_Adagrad(double **X, double **a,double **h, int  * sl, double ** Theta, double ** v_mom, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask);
void BP_MBatch_SGD_RMSprop(double **X, double **a,double **h, int  * sl, double ** Theta, double ** RMS_g, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask, int gamma_coef);
void BP_MBatch_SGD_Adadelta(double **X, double **a,double **h, int  * sl, double ** Theta, double ** RMS_g, double ** RMS_theta, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask, int gamma_coef);
void BP_MBatch_SGD_Adam(double **X, double **a,double **h, int  * sl, double ** Theta, double ** mt, double ** vt, double (*hidden_unit)(double),
		double (*output_unit)(double, double), int size_sample,int batch_size, int * index, int ** mask, int idx);
#endif /* NN_SPEC_H_ */
