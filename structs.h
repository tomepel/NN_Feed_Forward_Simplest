/*
 * structs.h
 *
 *  Created on: 2 mars 2017
 *      Author: tepelbaum
 */

#ifndef STRUCTS_H_
#define STRUCTS_H_

struct Neural_Network_tag;



typedef struct Neural_Network_tag{
int *   s_l; 			// the sizes of the hidden layers
double  * X; 			// the tidy data, of size num_file*(N_row*T_size_training_dat)*(s_input+s_output), contain both input and output
double  ** X_train; 	// the training set
double  ** X_cv; 		// the cross validation set
double  ** X_test; 		// the test set
double  ** Theta;		// The weight matrix Theta^(l)_ij, of size (s_l+1)*(s_lp1)
double  ** delta;		// for back propagation purposes
double  ** a_train;			// the linear combination at each layer : a^(l) = Theta^(l-1).h^(l-1), of size s_l
double  ** h_train;			// the activation variables, h^(l) = g( a^(l) ), with activation function (tanh, logistic, relu...)
double  ** a_cv;			// the linear combination at each layer : a^(l) = Theta^(l-1).h^(l-1), of size s_l
double  ** h_cv;			// the activation variables, h^(l) = g( a^(l) ), with activation function (tanh, logistic, relu...)
double  ** a_test;			// the linear combination at each layer : a^(l) = Theta^(l-1).h^(l-1), of size s_l
double  ** h_test;			// the linear combination at each layer : a^(l) = Theta^(l-1).h^(l-1), of size s_l
}Neural_Network;



#endif /* STRUCTS_H_ */
