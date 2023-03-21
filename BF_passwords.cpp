#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <thread>
#include "BF_passwords.hpp"

std::random_device rd;
std::mt19937 gen(rd());

void CharSet::make_char_set() {
    // Assign numbers
    for(int i = 0; i < 10; i++) {
        numbers[i] = 48 + i;
    }

    // Assign alphabets
    for(int i = 0; i < 26; i++) {
        alphabet[i] = 65 + i;
        alphabet[26 + i] = 97  + i;
    }

    // Assign numbers and alphabets
    std::copy(numbers, numbers + 10, numAndAlpha);
    std::copy(alphabet, alphabet + 52, numAndAlpha + 10);

    // Assign numbers and special characters
    std::copy(numbers, numbers + 10, numAndSpecial);
    std::copy(special, special + 32, numAndSpecial + 10);

    // Assign alphabets and special characters
    std::copy(alphabet, alphabet + 52, alphaAndSpecial);
    std::copy(special, special + 32, alphaAndSpecial + 52);

    // Assign all the characters
    std::copy(numbers, numbers + 10, allChar);
    std::copy(alphabet, alphabet + 52, allChar + 10);
    std::copy(special, special + 32, allChar + 62);
}

CharSet::CharSet() { // Constructor of the class Charset
    make_char_set();
}

void CharSet::make_password() {
    make_num_password();
    make_alpha_password();
    make_special_password();
    make_num_and_alpha_password();
    make_num_and_special_password();
    make_alpha_and_special_password();
    make_all_char_password();
}

void CharSet::make_num_password() { // Make 5~8 letter passwords with numbers

    std::uniform_int_distribution<int> numRange(0, 9);
    int passwordSetSize = 0;

    for(int i = 4; i < 9; i++) { // Password length from 4 to 8
        for(int k = 0; k < 10; k++) { // 10 passwords for each length
            std::string password;

            for(int j = 0; j < i; j++) { // Generate new password for each given length
                password += numbers[numRange(gen)];
            }
            
            passwordSetSize = passwordsHash.size();
            passwordsHash.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == passwordsHash.size()) { // Check if there was a duplicated password
                k -= 1; // Regenerate a password
            } 
        }
    }
}

void CharSet::make_alpha_password() { // Make 5~8 letter passwords with alphabets

    std::uniform_int_distribution<int> alphaRange(0, 51);
    int passwordSetSize = 0;
    
    for(int i = 4; i < 9; i++) { // Password length from 4 to 8
        for(int k = 0; k < 10; k++) { // 10 passwords for each length
            std::string password;

            for(int j = 0; j < i; j++) { // Generate new password for each given length
                password += alphabet[alphaRange(gen)];
            }

            passwordSetSize = passwordsHash.size();
            passwordsHash.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == passwordsHash.size()) { // Check if there was a duplicated password
                k -= 1; // Regenerate a password
            }
        }
    }
}

void CharSet::make_special_password() { // Make 5~8 letter passwords with special letters

    std::uniform_int_distribution<int> specialRange(0, 31);
    int passwordSetSize = 0;
    
    for(int i = 4; i < 9; i++) { // Password length from 4 to 8
        for(int k = 0; k < 10; k++) { // 10 passwords for each length
            std::string password;

            for(int j = 0; j < i; j++) { // Generate new password for each given length
                password += special[specialRange(gen)];
            }

            passwordSetSize = passwordsHash.size();
            passwordsHash.insert(password); // Insert newly generated password into pre-defined hash

            if(passwordSetSize == passwordsHash.size()) { // Check if there was a duplicated password
                k -= 1; // Regenerate a password
            }
        }
    }
}

void CharSet::make_num_and_alpha_password() { // Make 5~8 letter passwords with numbers and alphabets

    std::uniform_int_distribution<int> numAndAlphaRange(0, 61);
    int passwordSetSize = 0;

    for(int i = 4; i < 9; i++) { // Password length from 4 to 86
        for(int k = 0; k < 10; k++) { // 10 passwords for each length
            std::string password;

            for(int j = 0; j < i; j++) { // Generate new password for each given length
                password += numAndAlpha[numAndAlphaRange(gen)];
            }

            if(check_single_word_password(password, 2)) { // Check if the password consists of single character type
                k -= 1; // Regenerate a password
            }
            else{
                passwordSetSize = passwordsHash.size();
                passwordsHash.insert(password); // Insert newly generated password into pre-defined hash
                if(passwordSetSize == passwordsHash.size()) { // Check if there was a duplicated password
                    k -= 1; // Regenerate a password
                }
            }
        }
    }
}

void CharSet::make_num_and_special_password() { // Make 5~8 letter passwords with numbers and special letters

    std::uniform_int_distribution<int> numAndSpecialRange(0, 41);
    int passwordSetSize = 0;

    for(int i = 4; i < 9; i++) { // Password length from 4 to 86
        for(int k = 0; k < 10; k++) { // 10 passwords for each length
            std::string password;

            for(int j = 0; j < i; j++) { // Generate new password for each given length
                password += numAndSpecial[numAndSpecialRange(gen)];
            }

            if(check_single_word_password(password, 2)) { // Check if the password consists of single character type
                k -= 1; // Regenerate a password
            }
            else{
                passwordSetSize = passwordsHash.size();
                passwordsHash.insert(password); // Insert newly generated password into pre-defined hash
                if(passwordSetSize == passwordsHash.size()) { // Check if there was a duplicated password
                    k -= 1; // Regenerate a password
                }
            }
        }
    }
}

void CharSet::make_alpha_and_special_password() { // Make 5~8 letter passwords with alphabets and special letters

    std::uniform_int_distribution<int> alphaAndSpecialRange(0, 83);
    int passwordSetSize = 0;

    for(int i = 4; i < 9; i++) { // Password length from 4 to 86
        for(int k = 0; k < 10; k++) { // 10 passwords for each length
            std::string password;

            for(int j = 0; j < i; j++) { // Generate new password for each given length
                password += alphaAndSpecial[alphaAndSpecialRange(gen)];
            }

            if(check_single_word_password(password, 2)) { // Check if the password consists of single character type
                k -= 1; // Regenerate a password
            }
            else{
                passwordSetSize = passwordsHash.size();
                passwordsHash.insert(password); // Insert newly generated password into pre-defined hash
                if(passwordSetSize == passwordsHash.size()) { // Check if there was a duplicated password
                    k -= 1; // Regenerate a password
                }
            }
        }
    }
}

void CharSet::make_all_char_password() { // Make 5~8 letter passwords with the combination of numbers, alphabets and special letters

    std::uniform_int_distribution<int> allCharRange(0, 93);
    int passwordSetSize = 0;

    for(int i = 4; i < 9; i++) { // Password length from 4 to 86
        for(int k = 0; k < 10; k++) { // 10 passwords for each length
            std::string password;

            for(int j = 0; j < i; j++) { // Generate new password for each given length
                password += allChar[allCharRange(gen)];
            }

            if(check_single_word_password(password, 3)) { // Check if the password consists of single character type
                k -= 1; // Regenerate a password
            }
            else{
                passwordSetSize = passwordsHash.size();
                passwordsHash.insert(password); // Insert newly generated password into pre-defined hash
                if(passwordSetSize == passwordsHash.size()) { // Check if there was a duplicated password
                    k -= 1; // Regenerate a password
                }
            }
        }
    }
}

bool check_single_word_password(std::string _password, int _combiNum) { // Function for checking if a password consisted of one single type of character
    
    int cntNum = 0;
    int cntAlpha = 0;
    int cntSpecial = 0;

    for(int i = 0; i < _password.size(); i++) {
        if(_password[i] >= '0' && _password[i] <= '9') { // Count numbers in a password
            cntNum++;
        }
        else if((_password[i] >= 'A' && _password[i] <= 'Z') || (_password[i] >= 'a' && _password[i] <= 'z')) { // Count alphabets in a password
            cntAlpha++;
        }
        else { // Count special letters in a password
            cntSpecial++;
        }
    }
    
    // Check if there is a count number same with the size of password, return true
    if(cntNum == _password.size() || cntAlpha == _password.size() || cntSpecial == _password.size()) {
        return true;
    }

    // Check if there is a count number same with 0 and 3 letter combination, return true
    if((cntNum == 0 || cntAlpha == 0 || cntSpecial == 0) && _combiNum == 3) {
        return true;
    }

    return false;
}