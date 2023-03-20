#pragma once
#include <iostream>
#include <string>
#include <set>
#include <random>

struct LengthCmp {
	bool operator() (const std::string &_Left, const std::string &_Right) const {
        if (_Left.size() == _Right.size())
			return _Left < _Right;
		else
			return _Left.size() < _Right.size();
	}
};

class CharSet{
private:
    char numbers[10];
    char alphabet[52];
    char special[32] = {'!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', 
                        '+', ',', '-', '.', '/', ':', ';', '<', '=' , '>', 
                        '?', '@', '[', '\\', ']', '^', '_', '`', '{', '}',
                        '|', '~'};
    char numAndAlpha[62];
    char numAndSpecial[41];
    char alphaAndSpecial[84];
public:
    char allChar[94];
    std::set<std::string, LengthCmp> passwordsHash;
    CharSet();
    void make_char_set();
    void make_password();
    void make_num_password();
    void make_alpha_password();
    void make_special_password();
    void make_num_and_alpha_password();
    void make_num_and_special_password();
    void make_alpha_and_special_password();
    void make_all_char_password();
};

bool check_single_word_password(std::string _password);