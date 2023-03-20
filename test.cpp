#include <iostream>
#include <unordered_set>
#include <string>

std::unordered_set<std::string> password;

int main() {

    password.insert("new");
    password.insert("new");

    for(auto i : password) {
        std::cout << '[' << i << ']' << std:: endl;
    }

    std::cout << "size: " << password.size() << std:: endl;
}