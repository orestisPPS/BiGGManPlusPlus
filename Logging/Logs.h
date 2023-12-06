//
// Created by hal9000 on 11/21/23.
//

#ifndef UNTITLED_LOGS_H
#define UNTITLED_LOGS_H

#include <memory>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "Timer.h"
#include "vector"
using namespace std;

class Logs {
    
    public:
        explicit Logs(string logName = "");
        
        void addComment(string comment);
        
        //==========================================Timers==========================================
        
        void startSingleObservationTimer(const string &logName);
        
        void stopSingleObservationTimer(const string &logName);
        
        const chrono::duration<double>& getSingleObservationDuration(const string &logName);
        
        void startMultipleObservationsTimer(const string &logName);
        
        void stopMultipleObservationsTimer(const string &logName);
        
        const list<chrono::duration<double>>& getMultipleObservationsDurations(const string &logName);
        
        //==========================================Data==========================================
        
        void setSingleObservationLogData(const std::string &logName, double value);
        
        void setMultipleObservationsLogData(const std::string &logName, double value);
        
        void setMultipleObservationsLogData(const std::string &logName, const list<double> &values);

        list<double> getMultipleObservationsLogData(const std::string &logName);
        
        void storeAndResetCurrentLogs();
        
        void storeAndResetAllLogs();
        
        void exportToCSV(const string &fileName, const string &filePath);
        
        
    private:
        
        unique_ptr<map <string, Timer>> _currentSingleObservationTimers;
        
        unique_ptr<map<string, list<Timer>>> _currentMultipleObservationTimers;
        
        unique_ptr<map<string, list<chrono::duration<double>>>> _singleObservationTimers;     
        
        unique_ptr<map<string, list<list<chrono::duration<double>>>>> _multipleObservationTimers;
        
        unique_ptr<map<string, list<double>>> _singleObservationData;

        unique_ptr<map<string, list<list<double>>>> _multipleObservationData;
        
        unique_ptr<string> _comments;
        
        string _logName;
        
};


#endif //UNTITLED_LOGS_H