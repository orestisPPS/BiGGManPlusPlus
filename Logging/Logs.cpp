//
// Created by hal9000 on 11/21/23.
//

#include "Logs.h"
#include "../LinearAlgebra/Array/Array.h"


Logs::Logs(std::string logName) :   _logName(std::move(logName)),
                                    _currentSingleObservationTimers(make_unique<map<string, Timer>>()),
                                    _currentMultipleObservationTimers(make_unique<map<string, list<Timer>>>()),
                                    _singleObservationData(make_unique<map<string, list<double>>>()),
                                    _multipleObservationData(make_unique<map<string, list<list<double>>>>()),
                                    _singleObservationTimers(make_unique<map<string, list<chrono::duration<double>>>>()),
                                    _multipleObservationTimers(make_unique<map<string, list<list<chrono::duration<double>>>>>()),
                                    _comments(make_unique<string>()) {
}

void Logs::addComment(string comment) {
    _comments->append(comment);
}

void Logs::startSingleObservationTimer(const std::string &logName) {

    if (_currentSingleObservationTimers->find(logName) == _currentSingleObservationTimers->end())
        _currentSingleObservationTimers->insert(make_pair(logName, Timer(logName)));
    else
        _currentSingleObservationTimers->at(logName) = Timer(logName);
    _currentSingleObservationTimers->at(logName).start();
}

void Logs::stopSingleObservationTimer(const std::string &logName) {
    if (_currentSingleObservationTimers->find(logName) == _currentSingleObservationTimers->end())
        throw std::runtime_error("Logs::stopSingleObservationTimer: Timer " + logName + " does not exist.");
    
    _currentSingleObservationTimers->at(logName).stop();
    if (_singleObservationTimers->find(logName) == _singleObservationTimers->end())
        _singleObservationTimers->emplace(logName, list<chrono::duration<double>>());
    _singleObservationTimers->at(logName).push_back(_currentSingleObservationTimers->at(logName).duration());
}

const chrono::duration<double>& Logs::getSingleObservationDuration(const std::string &logName) {
    if (_singleObservationTimers->find(logName) == _singleObservationTimers->end())
        throw std::runtime_error("Logs::getSingleObservationDuration: Timer " + logName + " does not exist.");
    
    return _singleObservationTimers->at(logName).back();
}

void Logs::startMultipleObservationsTimer(const std::string &logName) {
    if (_currentMultipleObservationTimers->find(logName) == _currentMultipleObservationTimers->end())
        _currentMultipleObservationTimers->insert(make_pair(logName, list<Timer>()));
    auto timer = Timer(logName);
    timer.start();
    _currentMultipleObservationTimers->at(logName).push_back(timer);
}

void Logs::stopMultipleObservationsTimer(const std::string &logName) {
    if (_currentMultipleObservationTimers->find(logName) == _currentMultipleObservationTimers->end())
        throw std::runtime_error("Logs::stopMultipleObservationsTimer: Timer " + logName + " does not exist.");
    
    _currentMultipleObservationTimers->at(logName).back().stop();
}

const list<chrono::duration<double>>& Logs::getMultipleObservationsDurations(const std::string &logName) {
    if (_multipleObservationTimers->find(logName) == _multipleObservationTimers->end())
        throw std::runtime_error("Logs::getMultipleObservationsDurations: Timer " + logName + " does not exist.");
    
    return _multipleObservationTimers->at(logName).back();
}

void Logs::storeAndResetCurrentLogs() {
    for (auto &timerPair : *_currentMultipleObservationTimers) {
    }
    _currentMultipleObservationTimers->clear();
    
    for (auto &data: *_multipleObservationData) {
       data.second.emplace_back();   
    }
}

void Logs::setSingleObservationLogData(const std::string &logName, double value) {
    if (_singleObservationData->find(logName) == _singleObservationData->end())
        _singleObservationData->emplace(logName, list<double>());
    _singleObservationData->at(logName).push_back(value);
}


void Logs::setMultipleObservationsLogData(const std::string &logName, double value) {
    // Check if logName exists in the map
    if (_multipleObservationData->find(logName) == _multipleObservationData->end()) {
        _multipleObservationData->emplace(logName, list<list<double>>{list<double>()});
    }
    _multipleObservationData->at(logName).back().push_back(value);
}

void Logs::setMultipleObservationsLogData(const std::string &logName, const list<double> &values) {
    if (_multipleObservationData->find(logName) == _multipleObservationData->end()) {
        _multipleObservationData->emplace(logName, list<list<double>>());
    }
    // Add the new vector of values to the list
    _multipleObservationData->at(logName).push_back(values);
}


list<double> Logs::getMultipleObservationsLogData(const std::string &logName) {
    if (_multipleObservationData->find(logName) == _multipleObservationData->end())
        throw std::runtime_error("Logs::getMultipleObservationsLogData: Data " + logName + " does not exist.");
    
    return _multipleObservationData->at(logName).back();
}

void Logs::exportToCSV(const string &filePath, const string &fileName) {
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);

    // Create a timestamp string
    std::ostringstream timestamp;
    timestamp << std::put_time(&now_tm, "%d%m%Y_%H%M%S");

    // Create a unique filename with timestamp
    std::string filename = filePath + "/" + fileName + "_" + timestamp.str() + ".csv";

    std::ofstream file(filename);

    // Check if file is opened successfully
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    file << std::scientific << std::setprecision(15);
    // Adding comments
    if (!_comments->empty()) {
        file << "#Region: Comments" << std::endl;
        for (const auto &comment : *_comments) {
            file << "# " << comment << std::endl;
        }
        file << std::endl;
    }
    // Adding single observation Data
    if (!_singleObservationData->empty()) {
        file << "#Region: Single Observation Data" << std::endl;
        for (const auto &pair : *_singleObservationData) {
            file <<"LogEntry:" << pair.first << std::endl;
            for (const auto &value : pair.second) {
                file << value << std::endl;
            }
            file << std::endl;
        }
    }

    unsigned index = 0;
    // Adding multiple observation Data
    file << "#Region: Multiple Observation Data" << std::endl;
    size_t maxLength;
    unsigned logIndex = 0;
    for (const auto& pair : *_multipleObservationData) {
        auto &logName = pair.first;
        list<list<double>> logLists = pair.second;
        maxLength = 0;
        for (const auto &totalCallsList: logLists) {
            maxLength = std::max(maxLength, totalCallsList.size());
        }
        auto dataMatrix = make_unique<LinearAlgebra::Array<double>>(logLists.size(), maxLength);
        for (const auto &totalCallsList: logLists) {
            for (const auto &value: totalCallsList) {
                dataMatrix->at(logIndex, index) = value;
                ++index;
            }
            ++logIndex;
            index = 0;
        }
        file << "LogEntry:" << logName << std::endl;
        for (size_t i = 0; i < maxLength; ++i) {
            for (size_t j = 0; j < logLists.size(); ++j) {
                file << dataMatrix->at(j, i);
                if (j != logLists.size() - 1) {
                    file << ",";
                }
            }
            file << std::endl;
        }
        file << std::endl;
    }
    
    // Adding single observation timers
    if (!_singleObservationTimers->empty()) {
        file << "#Region: Single Observation Timers" << std::endl;
        for (const auto &timerPair : *_singleObservationTimers) {
            file << timerPair.first << std::endl;
            for (const auto &timer : timerPair.second) {
                file << timer.count() << std::endl;
            }
        }
    }
    
    // Adding multiple observation timers
    if (!_multipleObservationTimers->empty()) {
        file << "#Region: Multiple Observation Timers" << std::endl;
        for (const auto &timerPair : *_multipleObservationTimers) {
            file << timerPair.first << std::endl;
            //For each timer in the list
            auto timersMatrix = vector<vector<chrono::duration<double>>>(timerPair.second.size());
            index = 0;
            for (const auto &listOfVectors : timerPair.second) {
                //For each timer in the list
                for (const auto &timer : listOfVectors) {
                    timersMatrix[index].push_back(timer);
                }
            }
            for (const auto &timerList : timersMatrix) {
                for (size_t i = 0; i < timerList.size(); ++i) {
                    file << timerList[i].count();
                    if (i != timerList.size() - 1) {
                        file << ",";
                    }
                }
                file << std::endl;
            }
        }
    }

    file.close();
}

