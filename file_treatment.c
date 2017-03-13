/*
 * file_treatment.c
 *
 *  Created on: 1 mars 2017
 *      Author: tepelbaum
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "macro.h"
#include "structs.h"

/////////////// Determine the number of rows in the csv file START ////////////////////
// require a filename
// require a (0,1) int (0 = no header, 1 = header)
int csv_num_line(char * file)
{
	FILE* stream = fopen(file, "r");
	char line[100000];
	if (HEADER == 1) fgets(line, 100000, stream);
	int compteur = 0;
	while (fgets(line, 100000, stream))
	{
		compteur +=1;
	}
	fclose(stream);
	return compteur;

}
/////////////// Determine the number of row in the csv file END ///////////////////////



/////////////// Determine the number of columns in the csv file START /////////////////
// require a filename
// require a (0,1) int (0 = no header, 1 = header)
int csv_num_column(char * file)
{
	FILE* stream = fopen(file, "r");
	char line[100000];
	if (HEADER == 1) fgets(line, 100000, stream);
	int compteur = 0;
	fgets(line, 100000, stream);
	char* tmp = strdup(line);
    char* tok;
    for (tok = strtok(line, ";");tok && *tok;tok = strtok(NULL, ";\n"))
    {
    	compteur+=1;
    }
	free(tmp);
	free(tok);
	fclose(stream);
	return compteur;
}
/////////////// Determine the number of columns in the csv file END ///////////////////



////////////////////// Read the csv file START ////////////////////////////////////////
// require a filename
// require a matrix in which the file is stored
// require to know the number of columns (and implicitly the number of lines)
// require a (0,1) int (0 = no header, 1 = header)
void read_csv(char * file, double * X, int size)
{
	FILE* stream = fopen(file, "r");
	char line[100000];
	if (HEADER == 1) fgets(line, 100000, stream);
	int i = 0;
	int j;
	while (fgets(line, 100000, stream))
	{
		j = 1;
		char* tmp = strdup(line);
		char* tok;
		for (tok = strtok(line, ";");tok && *tok;tok = strtok(NULL, ";\n"))
		{
			X[j-1+size*i] =  strtod (tok,NULL);
			//memcpy(&X+(j-1+size*i), strtod (tok,NULL),1*sizeof(double));
			j += 1;
		}
		i+=1;
		free(tmp);
		free(tok);
	}
	fclose(stream);
}
////////////////////// Read the csv file END //////////////////////////////////////////
