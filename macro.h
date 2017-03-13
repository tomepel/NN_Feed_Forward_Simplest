/*
 * macro.h
 *
 *  Created on: 1 mars 2017
 *      Author: tepelbaum
 */

///////////////////////////////// Number of layers START /////////////////////////////////////////////////////////////////////
#define L_layer 4																											//
///////////////////////////////// Number of layers END ///////////////////////////////////////////////////////////////////////

///////////////////////////////// some constant START ////////////////////////////////////////////////////////////////////////
#define twothird 		0.6666666666666667 																				    //
#define fourthird 		1.3333333333333333 																				    //
#define tanh_m_twothird 	1.7159047085755392 																			    //
/////////////////////////////////some constant END  //////////////////////////////////////////////////////////////////////////

///////////////////////////////// Size of layers START ///////////////////////////////////////////////////////////////////////
#define s_input 12																											//
#define s_output 8																											//
#define s_hidden 10																											//
///////////////////////////////// Size of layers END /////////////////////////////////////////////////////////////////////////

///////////////////////////////// number of 1/4h considered START ////////////////////////////////////////////////////////////
#define N_1_over_4_h 72	// All quarter hour from 5 am to 11 pm																//
#define T_size_training_dat 60																								//
#define start_index 101	// Start at 5 am																					//
#define end_index 460	// Stop at 11 pm																					//
#define N_Neighbours 3																										//
///////////////////////////////// number of 1/4h considered END //////////////////////////////////////////////////////////////


///////////////////////////////// number of files and days START /////////////////////////////////////////////////////////////
#define N_FILE 10 																											//
#define N_ARC 396 																											//
#define HEADER 1																											//
#define size_training 8 * N_FILE / 10																						//
#define size_cv 1 * N_FILE / 10																								//
#define size_test 1 * N_FILE / 10																							//
#define size_file N_ARC*T_size_training_dat //(num_arc times T_size_training_dat) 											//
#define size_train_file size_file*size_training																				//
#define size_cv_file size_file*size_cv																						//
#define size_test_file size_file*size_test																					//
#define size_tot_X size_file*(s_output+s_input)																				//
#define S_DAY 602																											//
///////////////////////////////// Size of layers END /////////////////////////////////////////////////////////////////////////


///////////////////////////////// parameters for gradient descent START //////////////////////////////////////////////////////
#define ETA 0.001 // for gradient descent																					//
#define LAMBDA 0 // Regularization 																							//
#define GAMMA 0.9 // for momentum 																							//
#define EPSILON 1.0e-8																										//
#define BETA_1 0.9																											//
#define BETA_2 0.999																										//
///////////////////////////////// parameters for gradient descent END ////////////////////////////////////////////////////////


///////////////////////////////// parameters for evaluation model START //////////////////////////////////////////////////////
#define INDEX_ARC 7																											//
///////////////////////////////// parameters for evaluation model END ////////////////////////////////////////////////////////
