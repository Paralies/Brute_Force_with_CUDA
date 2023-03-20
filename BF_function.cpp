#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <thread>
//#include <unistd>
#include "BF_header.hpp"

void charSet::make_char_set() {
    for(int i = 0; i < 10; i++) {
        numbers[i] = 48 + i;
    }

    for(int i = 0; i < 26; i++) {
        alphabet[i] = 65 + i;
    }

    for(int i = 26; i < 52; i++) {
        alphabet[i] = 97 + i;
    }
}

charSet::charSet() { // Constructor of the class Charset
    make_char_set();
}

void charSet::make_num_password() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> numRange(0, 9);

    std::string password;

    for(int i = 4; i < 9; i++) { // Password length from 4 to 8
        for(int j = 0; j < i; i++) { // Generate new password for each given length
            password += numbers[numRange(gen)];
        }

        passwords.insert(password); // Insert newly generated password into pre-defined hash
    }
}

void charSet::make_alpha_password() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> alPhaRange(0, 51);

    std::string password;
    int passwordSetSize = 0;
    
    for(int i = 4; i < 9; i++) { // Password length from 4 to 8
        for(int k = 0; k < 10; k ++) { // 10 passwords for each length
            for(int j = 0; j < i; i++) { // Generate new password for each given length
                password += alphabet[alPhaRange(gen)];
            }

            passwordSetSize = password.size();
            passwords.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == password.size()) { // Check if there was a duplicated password
                k -= 1;
            }
        }
    }
}

void charSet::make_special_password() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> specialRange(0, 31);

    std::string password;
    int passwordSetSize = 0;
    
    for(int i = 4; i < 9; i++) { // Password length from 4 to 8
        for(int k = 0; k < 10; k ++) { // 10 passwords for each length
            for(int j = 0; j < i; i++) { // Generate new password for each given length
                password += special[specialRange(gen)];
            }

            passwordSetSize = password.size();
            passwords.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == password.size()) { // Check if there was a duplicated password
                k -= 1;
            }
        }
    }
}

void charSet::make_num_and_alpha_password() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> numAndAlphaRange(0, 61);

    std::string password;
    int passwordSetSize = 0;
    int numAndAlpha[62];
    
    std::copy(numbers, numbers + sizeof(numbers) / sizeof(numbers[0]), numAndAlpha);
    std::copy(alphabet, alphabet + sizeof(alphabet) / sizeof(alphabet[0]), numAndAlpha + sizeof(numbers) / sizeof(numbers[0]));

    for(int i = 4; i < 9; i++) { // Password length from 4 to 86
        for(int k = 0; k < 10; k ++) { // 10 passwords for each length
            for(int j = 0; j < i; i++) { // Generate new password for each given length
                password += numAndAlpha[numAndAlphaRange(gen)];
            }

            passwordSetSize = password.size();
            passwords.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == password.size()) { // Check if there was a duplicated password
                k -= 1;
            }
        }
    }
}

void charSet::make_num_and_special_password() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> numAndAlphaRange(0, 61);

    std::string password;
    int passwordSetSize = 0;
    int numAndAlpha[62];
    
    std::copy(numbers, numbers + sizeof(numbers) / sizeof(numbers[0]), numAndAlpha);
    std::copy(alphabet, alphabet + sizeof(alphabet) / sizeof(alphabet[0]), numAndAlpha + sizeof(numbers) / sizeof(numbers[0]));

    for(int i = 4; i < 9; i++) { // Password length from 4 to 86
        for(int k = 0; k < 10; k ++) { // 10 passwords for each length
            for(int j = 0; j < i; i++) { // Generate new password for each given length
                password += numAndAlpha[numAndAlphaRange(gen)];
            }

            passwordSetSize = password.size();
            passwords.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == password.size()) { // Check if there was a duplicated password
                k -= 1;
            }
        }
    }
}

void charSet::make_alpha_and_special_password() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> numAndAlphaRange(0, 61);

    std::string password;
    int passwordSetSize = 0;
    int numAndAlpha[62];
    
    std::copy(numbers, numbers + sizeof(numbers) / sizeof(numbers[0]), numAndAlpha);
    std::copy(alphabet, alphabet + sizeof(alphabet) / sizeof(alphabet[0]), numAndAlpha + sizeof(numbers) / sizeof(numbers[0]));

    for(int i = 4; i < 9; i++) { // Password length from 4 to 86
        for(int k = 0; k < 10; k ++) { // 10 passwords for each length
            for(int j = 0; j < i; i++) { // Generate new password for each given length
                password += numAndAlpha[numAndAlphaRange(gen)];
            }

            passwordSetSize = password.size();
            passwords.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == password.size()) { // Check if there was a duplicated password
                k -= 1;
            }
        }
    }
}

void charSet::make_all_char_password() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> numAndAlphaRange(0, 61);

    std::string password;
    int passwordSetSize = 0;
    int numAndAlpha[62];
    
    std::copy(numbers, numbers + sizeof(numbers) / sizeof(numbers[0]), numAndAlpha);
    std::copy(alphabet, alphabet + sizeof(alphabet) / sizeof(alphabet[0]), numAndAlpha + sizeof(numbers) / sizeof(numbers[0]));

    for(int i = 4; i < 9; i++) { // Password length from 4 to 86
        for(int k = 0; k < 10; k ++) { // 10 passwords for each length
            for(int j = 0; j < i; i++) { // Generate new password for each given length
                password += numAndAlpha[numAndAlphaRange(gen)];
            }

            passwordSetSize = password.size();
            passwords.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == password.size()) { // Check if there was a duplicated password
                k -= 1;
            }
        }
    }
}

bool check_same_consitent_password(std::string _password){
    
    int cntNum = 0;
    int cntAlpha = 0;
    int cntSpecial = 0;

    for(int i = 0; i < _password.size(); i++) {
        
    }

    if(cntNum == _password.size() || cntAlpha == _password.size() || cntSpecial == _password.size()) {
        return true;
    }

    return false;
}