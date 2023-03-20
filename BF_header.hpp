#include <iostream>
#include <string>
#include <unordered_set>

std::unordered_set<std::string> passwords;

class charSet{

    char numbers[10];
    char alphabet[52];
    char special[32] = {'!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', 
                        '+', ',', '-', '.', '/', ':', ';', '<', '=' , '>', 
                        '?', '@', '[', '\\', ']', '^', '_', '`', '{', '}',
                        '|', '~'};

    void make_char_set();
    void make_num_password();
    void make_alpha_password();
    void make_special_password();
    void make_num_and_alpha_password();
    void make_num_and_special_password();
    void make_alpha_and_special_password();
    void make_all_char_password();
};