#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <functional>
#include <chrono>

using namespace std;

// Global Variables 

vector<vector<vector<double>>> weights; // weight[l][j][k] = weight associated with neuron j in layer [l] with neuron k in layer [l-1]
vector<vector<double>> biases; // bias[l][j] = bias for neuron j in layer [l]
vector<vector<double>> z;
vector<vector<vector<double>>> nabla_w;
vector<vector<double>> nabla_b;

vector<vector<vector<double>>> training_data;
vector<vector<vector<double>>> test_data;
vector<vector<vector<double>>> validation_data;

long long seed = chrono::system_clock::now().time_since_epoch().count()/10000000;
long long MOD = 1e9+7;
bool RNG_RAN = false;
// Mathematical Functions

double sigmoid(double x){
    return (1.0/double(1.0+exp(-1LL*x)));
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
    // cout<<"Extracting data from python objects..";
    // create_temp_files();
    cout<<"Importing data to c++ vectors..."<<endl;
    import_data("training_data.txt", training_data);
    import_data("test_data.txt", test_data);
    import_data("validation_data.txt", validation_data);
    // cout<<"Deleting temporary files..."<<endl;
    // delete_temp_files();
    // cout<<"Data Import Complete!!"<<endl;
}

double rng(){
    //Simple linear congruence based Pseudo Random Number Generator 
    if(!RNG_RAN){
        for(int i = 0;i<100;i++){
            seed = (7*seed + 9)%MOD;
        }
        RNG_RAN = true;
    }
    seed = (7*seed + 9)%MOD;
    return seed/double(MOD);
}

vector<double> box_muller(int n){
    // Generates n samples from standard normal N(0,1)

    vector<double> u1(n/2);
    vector<double> u2(n/2);
    for(auto &u : u1){
        u = rng();
    }
    for(auto &u : u2){
        u = rng();
    }
    vector<double> Z1(n/2);
    vector<double> Z2(n/2);
    for(int i = 0;i<n/2;i++){
        Z1[i] = sqrt((-2LL)*log(u1[i])) * cos(2*M_PI *u2[i]);
        Z2[i] = sqrt((-2LL)*log(u1[i])) * sin(2*M_PI *u2[i]);
    }

    vector<double> ret(n);
    int i = 0;
    for(int i = 0;i<n/2;i++){
        ret[i] = Z1[i];
    }
    for(int i = n/2;i-n/2<n/2;i++){
        ret[i] = Z2[i-n/2];
    }
    if(n&1){
        double u1_ex = rng();
        double u2_ex = rng();
        ret[n-1] = sqrt((-2)*log(u1_ex)) * cos(2*M_PI * u2_ex);
    }
    return ret;
}

void setNetwork(vector<int> &&layers){
    weights.clear();
    biases.clear();
    z.clear();
    for(int l = 0;l<layers.size();l++){
        //Set initial biases
        vector<double> bias = box_muller(layers[l]);
        vector<double> Z_temp(layers[l]);
        vector<double> nabla_bias(layers[l],0);
        //Set initial weights
        vector<vector<double>> weight;
        vector<vector<double>> nabla_weight;
        
        if(l != 0){
            weight.resize(layers[l], vector<double>(layers[l-1]));
            nabla_weight.resize(layers[l], vector<double>(layers[l-1],0));
        }
        
        for(vector<double> &a : weight){
            for(auto &b:  a){
                b = sqrt(1.0/(a.size()))*box_muller(1)[0];
            }
        }
        
        biases.push_back(bias);
        weights.push_back(weight);
        z.push_back(Z_temp);
        nabla_b.push_back(nabla_bias);
        nabla_w.push_back(nabla_weight);
    }
    cout << "Done setting the initial parameters for the Network!!" << endl;
}

void feedforward(vector<double> &input){
    int layers = biases.size();
    //Activations for first layer
    for(int j = 0;j < z[0].size(); j++){
        z[0][j] = input[j];
    }
    //Feedforward
    for(int l = 1;l<layers;l++){
        for(int j = 0;j<z[l].size();j++){
            double weighted_sum = 0;
            for(int k = 0;k < z[l-1].size(); k++){
                if(l == 1){
                    weighted_sum += weights[l][j][k]*(z[l-1][k]);    
                }
                else{
                    weighted_sum += weights[l][j][k]*sigmoid(z[l-1][k]);
                }
                
            }   
            z[l][j] = weighted_sum + biases[l][j];
        }
    }
}

int final_result(){
    int n = z.size();
    double mx = z[n-1][0];
    int mx_ind = 0;
    for(int i = 1;i<z[n-1].size();i++){
        if(z[n-1][i] > mx){
            mx = z[n-1][i];
            mx_ind = i;
        }
    }
    return mx_ind+1;
}

template <class T>
void fisher_bates_random_shuffle(vector<T> &x){
    function<int(int,int)> randRange = [&] (int lo, int hi) -> int{
        int multiplier = (hi-lo);
        int random_index = (int)(multiplier*rng()) + lo;
        return random_index;
    };
    for(int i = x.size()-1;i >= 0;i--){
        int j = randRange(0,i-1);
        swap(x[i],x[j]);
    }
}

void backprop(vector<vector<double>> pt){
    //Compute z values for each node using feedforward
    feedforward(pt[0]); //pt[0] is input and pt[1] is the expected output
    int L = weights.size();
    vector<vector<double>> delta(L);
    for(int l = 0;l<L;l++){
        delta[l].resize(biases[l].size());
    }
    //Calculate delta for last layer
    for(int j = 0; j <biases[L-1].size(); j++){
        double y = 0;
        if(j == pt[1][0]){
            y = 1;
        }
        delta[L-1][j] = (sigmoid(z[L-1][j]) - y)*sigmoid_prime(z[L-1][j]);
        nabla_b[L-1][j] = delta[L-1][j];
        for(int k = 0;k<nabla_w[L-1][j].size();k++){
            nabla_w[L-1][j][k] = sigmoid(z[L-2][k])*delta[L-1][j];
        }
    }
    
    //Stepwise for each layer backwards: 
    for(int l = L-2; l>=0; l--){
        //  Calculate delta of layer and compute nabla_b[l][j] and nabla_w[l][j][k]
        for(int j = 0; j < biases[l].size(); j++){
            delta[l][j] = 0;
            for(int i = 0;i<weights[l+1].size();i++){
                delta[l][j] += weights[l+1][i][j]*delta[l+1][i];
            }
            delta[l][j] *= sigmoid_prime(z[l][j]);
            nabla_b[l][j] += delta[l][j];
            if(l == 0){
                continue;
            }
            for(int k = 0;k<weights[l][j].size();k++){
                nabla_w[l][j][k] += sigmoid(z[l-1][k])*delta[l][j];
            }
        }
    }
}

void update_batch(vector<vector<vector<double>>> &data, double eta, int mini_batch_size){
    //For each data point in batch, we must calculate dC/dw and dC/db using backprop
    //Then we average these values out over the entire batch and update weights and biases accordingly
    for (auto &layer : nabla_w) {
        for (auto &neuron : layer) {
            fill(neuron.begin(), neuron.end(), 0);
        }
    }
    for (auto &layer : nabla_b) {
        fill(layer.begin(), layer.end(), 0);
    }

    for(auto &pt : data){
        backprop(pt);
    }
    // cout<<nabla_b[0][0]<<endl;
    // cout<<"Ran backprop for all pts in batch"<<endl;
    for(int l = 0;l < weights.size(); l++){
        for(int j = 0;j < weights[l].size(); j++){
            for(int k = 0;k < weights[l][j].size(); k++){
                weights[l][j][k] = weights[l][j][k] - (eta/mini_batch_size)*nabla_w[l][j][k];
            }
        }
    }
    for(int l = 0;l < biases.size(); l++){
        for(int j = 0;j <biases[l].size();j++){
            biases[l][j] = biases[l][j] - (eta/mini_batch_size)*nabla_b[l][j];
        }
    }
    for(int j = 0;j<biases[biases.size()-1].size();j++){
        cout<<biases[biases.size()-1][j]<<" ";
    }cout<<endl;
}

void evaluate_performance();

void SGD(int epochs, int mini_batch_size, double learning_rate){
    double &eta = learning_rate;
    //Loop for Epochs
    for(int epoch = 0;epoch < epochs;epoch++){
        //Shuffle the training data randomly
        cout<<"Training commence for epoch : "<<epoch+1<<endl;
        fisher_bates_random_shuffle(training_data);
        //Divide into batches of size mini_batch_size
        vector<vector<vector<vector<double>>>> batches;
        for(int i = 0;i< (training_data.size()/mini_batch_size); i++){
            vector<vector<vector<double>>> temp(mini_batch_size);
            for(int j = 0;j<mini_batch_size; j++){
                temp[j] = training_data[i*mini_batch_size + j];
            }
            batches.emplace_back(temp);
        }
        
        //Run update for all mini batches
        for(auto &batch : batches){
            update_batch(batch,eta,mini_batch_size);
        }
        
        evaluate_performance();
    }
}

void evaluate_performance(){
    int total = test_data.size();
    int correct = 0;
    for(auto &pt: test_data){
        feedforward(pt[0]);
        if(final_result() == pt[1][0]) correct++;
    }
    cout<<correct<<"/"<<total<<" passed"<<endl;
}

int main(){
    import_data_wrapper();
    setNetwork({784,10,10});
    SGD(100, 1000, 10);
}