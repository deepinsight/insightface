//
// Created by Tunm-Air13 on 2024/3/22.
//

#ifndef INSPIREFACE_SIMPLE_CSV_WRITER_H
#define INSPIREFACE_SIMPLE_CSV_WRITER_H

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip> // Used to set the output format

class SimpleCSVWriter {
public:
    SimpleCSVWriter(const std::string& filepath) {
        // Check whether the file exists
        std::ifstream file(filepath);
        if (!file.good()) {
            // The file does not exist. Create a new csv file
            std::ofstream outfile(filepath);
            if (!outfile.is_open()) {
                std::cerr << "Failed to create file: " << filepath << std::endl;
            }
            outfile.close();
        }
        // Save the file path for later use
        this->filepath = filepath;
    }

    virtual ~SimpleCSVWriter() {} // Add a virtual destructor to ensure correct destructor behavior

protected:
    std::string filepath;

    void insertData(const std::vector<std::string>& data) {
        std::ofstream file(this->filepath, std::ios_base::app); // Open the file in append mode
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << this->filepath << std::endl;
            return;
        }
        for (const auto& datum : data) {
            file << datum;
            if (&datum != &data.back()) { // If it is not the last element, add a comma separation
                file << ",";
            }
        }
        file << "\n"; // Add a newline character after each inserted row of data
        file.close();
    }
};

class BenchmarkRecord : public SimpleCSVWriter {
public:
    BenchmarkRecord(const std::string& filepath, const std::string &name = "Benchmark") : SimpleCSVWriter(filepath) {
        std::ifstream file(this->filepath);
        if (file.peek() == std::ifstream::traits_type::eof()) { // If the file is empty, insert header data
            std::vector<std::string> header = {name, "Loops", "Total Time(ms)", "Average Time(ms)"};
            SimpleCSVWriter::insertData(header);
        }
    }

    void insertBenchmarkData(const std::string &caseName, int loops, double totalCost, double avgCost) {
        std::ofstream file(this->filepath, std::ios_base::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << this->filepath << std::endl;
            return;
        }
        // Format output
        file << std::fixed << std::setprecision(5);
        file << caseName << "," << loops << "," << totalCost << "," << avgCost << "\n";
        file.close();
    }
};


class EvaluationRecord : public SimpleCSVWriter {
public:
    EvaluationRecord(const std::string& filepath) : SimpleCSVWriter(filepath) {
        std::ifstream file(this->filepath);
        if (file.peek() == std::ifstream::traits_type::eof()) { // If the file is empty, insert header data
            std::vector<std::string> header = {"Resource Version", "Dataset", "Accuracy", "Best Threshold"};
            SimpleCSVWriter::insertData(header);
        }
    }

    void insertEvaluationData(const std::string &modelName, const std::string &dataset, double accuracy, double bestThreshold) {
        std::ofstream file(this->filepath, std::ios_base::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << this->filepath << std::endl;
            return;
        }
        // Format output
        file << std::fixed << std::setprecision(5);
        file << modelName << "," << dataset << "," << accuracy << "," << bestThreshold << "\n";
        file.close();
    }
};


#endif //INSPIREFACE_SIMPLE_CSV_WRITER_H
