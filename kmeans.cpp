/* ========================================================
 *   Copyright (C) 2014 All rights reserved.
 *
 *   filename : kmeans.cpp
 *   author   : guoxinpeng
 *   date     : 2014-09-25
 *   info     :  
 * ======================================================== */


#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <climits>
#include <string>

using namespace std;

#define LINE_LEN 256

int N, M, K, max_iter;
char * input_file;
char * output_dir;
vector< vector<double> > points;
vector< vector<double> > centroids;
vector< double > mius;
vector< double > sigmas;
vector< double > low_bounds;
vector< double > up_bounds;
vector< int > labels;
vector< char* > instances;  

void init_alldata(){
    srand(time(0));
    for(int i = 0; i < N; i++) labels.push_back(-1);
    for(int j = 0; j < M; j++) mius.push_back(0.0);
    for(int j = 0; j < M; j++) low_bounds.push_back(INT_MAX*1.0);
    for(int j = 0; j < M; j++) up_bounds.push_back(INT_MIN*1.0);
    for(int j = 0; j < M; j++) sigmas.push_back(0);
}

char** split(const char * string, char delim, int * count) {
    if( !string )
        return 0;
    int i, j, c;
    i = 0; j = c = 1;
    int length = strlen(string);
    char * copy_str = (char *) malloc(length+1);
    memmove(copy_str, string, length);
    copy_str[length] = '\0';
    for(; i < length; i++) {
        if(copy_str[i] == delim) {
            c += 1;
        }
    }
    (*count) = c;
    char** str_array = (char**)malloc(sizeof(char*) * c);
    str_array[0] = copy_str;
    for(i = 0; i < length; i++) {
        if(copy_str[i] == delim) {
            copy_str[i] = '\0';
            str_array[j++] = copy_str + i + 1;
        }
    }
    return str_array;
}

int load_data(){
    int sz;
    FILE * fp;
    if((fp = fopen(input_file, "r")) == NULL) {
        fprintf(stderr, "data file is not valid\n");
        return -1;
    }
    char buffer[LINE_LEN];
    char ** str_array = NULL;
    N = 0;
    while(fgets(buffer, LINE_LEN, fp) != NULL){
        vector<double> point_x;
        str_array = split(buffer,'\t',&sz);
        M = sz - 1;
        instances.push_back(str_array[0]);  
        for(int j = 1; j < sz; j++) {
            point_x.push_back(atof(str_array[j]));
        }
        points.push_back(point_x);
        N++;
    }
    return 0;
}

void normalize_data(){
    for(int j = 0; j < M; j++) {
        for(int i = 0; i < N; i++) {
            mius[j] += points[i][j];
        }
        mius[j] /= N;
    }
    for(int j = 0; j < M; j++) {
        for(int i = 0; i < N; i++) {
            sigmas[j] += (points[i][j]-mius[j]) * (points[i][j]-mius[j]);
        }
        sigmas[j] = sqrt(sigmas[j]/N);
    }
    for(int i = 0; i < N; i++) for(int j = 0; j < M; j++) {
        points[i][j]    = (points[i][j]-mius[j])/sigmas[j];
        low_bounds[j]   = min(low_bounds[j], points[i][j]);
        up_bounds[j]    = max(up_bounds[j], points[i][j]);
    }
}

void init_centroid(){
    for(int i = 0; i < K; i++) {
        vector< double > centroid_x;
        for(int j = 0; j < M; j++) {
            centroid_x.push_back(low_bounds[j] + (up_bounds[j]-low_bounds[j]) * (rand() * 1.0 / RAND_MAX));
        }
        centroids.push_back(centroid_x);
    }
}

void update_centroid(){
    vector< vector<double> > sum_centroids;
    for(int k = 0; k < K; k++) {
        vector< double > sum_centroid_x;
        for(int j = 0; j < M; j++) {
            sum_centroid_x.push_back(0.0);
        }
        sum_centroids.push_back(sum_centroid_x);
    }
    for(int i = 0; i < N; i++) for(int j = 0; j < M; j++)  {
        sum_centroids[labels[i]][j] += points[i][j];
    }
    for(int k = 0; k < K; k++) for(int j = 0; j < M; j++) {
        centroids[k][j] = sum_centroids[k][j] * 1.0 / N;
    }
}

void classify(){
    for(int i = 0; i < N; i++) {
        double min_dis = INT_MAX;
        int label_x = -1;
        for(int k = 0; k < K; k++) {
            double new_dis = 0.0;
            for(int j = 0; j < M; j++) {
                new_dis += (points[i][j]-centroids[k][j]) * (points[i][j]-centroids[k][j]);
            }
            if(min_dis > new_dis) {
                min_dis = new_dis;
                label_x = k;
            }
        }
        labels[i] = label_x;
    }
}

void output_classifier(){
    for(int k = 0; k < K; k++) {
        for(int j = 0; j < M; j++) {
            printf("%.3f\t", centroids[k][j]);
        }
        printf("\n");
    }
}

void output_pointclass(){
    FILE * ofp;
    ofp = fopen(output_dir, "w");
    if (ofp == NULL) {
        fprintf(stderr, "ERROR: Can't open output file %s!\n", output_dir);
        return ;
    }
    for(int i = 0; i < N; i++) {
        fprintf(ofp, "%s\t%d\n", instances[i], labels[i]+1);
    }
}

int command_line_parse(int argc, char * argv[]) {
    int i = 0;
    if(argc != 9) {
        fprintf(stderr, "ERROR: command line not well formatted\n");
        return -1;
    }
    while(i < argc) {
        char * arg = argv[i];
        if(strcmp(arg, "-k") == 0) { K = atoi(argv[++i]); }
        else if(strcmp(arg, "-n") == 0) { max_iter = atoi(argv[++i]); }
        else if(strcmp(arg, "-i") == 0) { input_file = argv[++i]; }
        else if(strcmp(arg, "-o") == 0) { output_dir = argv[++i]; }
        i++;
    }
    return 0;
}

/**
 * @brief: print the command line parameter tips
 **/
void print_help() {
    fprintf(stderr, "\n     Kmeans Command Usage:   \n\n");
    fprintf(stderr, "       ./kmeans -k <int> -n <int> -i <string> -o <string> \n\n");
    fprintf(stderr, "       -k  cluster number              \n");
    fprintf(stderr, "       -n  max iterators               \n");
    fprintf(stderr, "       -i  input data file             \n");
    fprintf(stderr, "       -o  output dir                  \n\n");
}

/**
 * @brief: main function for kmeans
 **/
int main(int argc, char * argv[]) {
    if(command_line_parse(argc, argv) != 0) { print_help(); return -1; }
    if(load_data() != 0) { print_help(); return -1; }
    init_alldata();
    normalize_data();
    init_centroid();
    for(int iter = 0; iter < max_iter; iter++){
        classify();
        update_centroid();
    }
    output_pointclass();
    return 0;
}
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
