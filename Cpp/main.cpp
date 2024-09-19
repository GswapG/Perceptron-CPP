#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;

// Global Variables 

vector<vector<vector<double>>> weight; // weight[l][j][k] = weight associated with neuron j in layer [l] with neuron k in layer [l-1]
vector<vector<double>> bias; // bias[l][j] = bias for neuron j in layer [l]

vector<vector<vector<double>>> training_data;
vector<vector<vector<double>>> test_data;
vector<vector<vector<double>>> validation_data;

// Mathematical Functions

double sigmoid(double x){
    return (1/(1+exp(-1LL*x)));
}

double sigmoid_prime(double x){
    return sigmoid(x)*(1-sigmoid(x));
}

//Utility functions 

void import_data(string path, vector<vector<vector<double>>> &vec){
    ifstream file("../../" + path);
    string line;
    if(!file){
        cout<<"Error opening file : "<<path<<endl; 
    }
    int line_number = 0; // line_number is data point number 
    while(getline(file, line)){
        string temp = "";
        vector<vector<double>> in_out_pair;
        vector<double> input;
        vector<double> output;
        for(char c : line){
            if(c != ' '){
                temp.push_back(c);
            }
            else{
                input.push_back(stod(temp));
                temp = "";
            }
        }
        output.push_back(stod(temp));
        in_out_pair.push_back(input);
        in_out_pair.push_back(output);
        vec.push_back(in_out_pair);
    }
}

void delete_temp_files(){
    remove("../../training_data.txt");
    remove("../../test_data.txt");
    remove("../../validation_data.txt");
}

void create_temp_files(){
    system("python ../../dataWriter.py");
}

void import_data_wrapper(){
    cout<<"Extracting data from python objects..";
    create_temp_files();
    cout<<"Importing data to c++ vectors..."<<endl;
    import_data("training_data.txt", training_data);
    import_data("test_data.txt", test_data);
    import_data("validation_data.txt", validation_data);
    cout<<"Deleting tempory files..."<<endl;
    delete_temp_files();
    cout<<"Data Import Complete!!"<<endl;
}

int main(){
    import_data_wrapper();
}