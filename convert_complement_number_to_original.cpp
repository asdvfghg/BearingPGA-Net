#include <iostream>
#include <bitset>
#include <fstream>
#include <string>
#include <dirent.h>


using namespace std;



short complement_to_original_preserve_sign(short x) {
    short sign = (x >> 15) & 1;
    short original = sign << 15;
    if (sign == 1) {
        x = ~x + 1;
        original |= x;
    }
    else {
        original = x;
    }

    return original;
}

int main() {

    // Open  complementary code
    ifstream infile("./Weight_Parameters/Fixed_Point/scnn_layer_2.txt");
    if (!infile) {
        cout << "Failed to open input file." << endl;
        return 1;
    }

    // Create original code
    ofstream outfile("./Weight_Parameters/Fixed_Point/scnn_layer_2_new.txt");
    if (!outfile) {
        cout << "Failed to create output file." << endl;
        return 1;
    }




    string line;
    while (getline(infile, line)) {
        short x = stoi(line, nullptr, 2);
        short original = complement_to_original_preserve_sign(x);
        outfile << bitset<16>(original) << endl;
    }

    infile.close();
    outfile.close();

    cout << "Conversion finished." << endl;
    return 0;
}



