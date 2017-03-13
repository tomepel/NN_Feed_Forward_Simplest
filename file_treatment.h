/*
 * file_treatment.h
 *
 *  Created on: 1 mars 2017
 *      Author: tepelbaum
 */



#ifndef FILE_TREATMENT_H_
#define FILE_TREATMENT_H_



int csv_num_line(char * file);
int csv_num_column(char * file);
void read_csv(char * file, double * X, int size);



#endif /* FILE_TREATMENT_H_ */
