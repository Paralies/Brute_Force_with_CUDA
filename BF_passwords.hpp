#pragma once
#include <iostream>
#include <string>
#include <set>
#include <random>

struct LengthCmp { // Sort from small size to big size, from little ascii to big ascii
	bool operator() (const std::string &_Left, const std::string &_Right) const {
        if (_Left.size() == _Right.size())
			return _Left < _Right; // from small size to big size
		else
			return _Left.size() < _Right.size(); // from little ascii to big ascii
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
    std::set<std::string, LengthCmp> passwordsHash; // Store password in set type container to prevent save duplicated one
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