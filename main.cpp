#include <iostream>
#include <string>
#include <fstream>
#include "BF_passwords.hpp"

int main() {

    std::ofstream logFile;
    logFile.open("bruteForce.log");

    std::ofstream passwordFile;
    passwordFile.open("password.txt");

    CharSet passwordBF;
    passwordBF.make_password();

    for(auto i : passwordBF.passwordsHash) {
        passwordFile << i << std::endl;
        //std::cout << i << std::endl;
    }
    
    logFile.close();
    passwordFile.close();
}