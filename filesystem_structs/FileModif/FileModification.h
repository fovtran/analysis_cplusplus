// cl /experimental:module /EHsc /MD /std:c++latest FileTypeMap.cxx FileModificationProgram.cxx
#pragma once

#include <time.h>
#include <string>
#include <cwchar>
#include <vector>
#include <map>

#include <Windows.h>

import std.core;
using std::wstring;
using std::vector;
using std::map;


using namespace std;
/* constants defined in winnt.h :
#define FILE_ACTION_ADDED                   0x00000001
#define FILE_ACTION_REMOVED                 0x00000002
#define FILE_ACTION_MODIFIED                0x00000003
#define FILE_ACTION_RENAMED_OLD_NAME        0x00000004
#define FILE_ACTION_RENAMED_NEW_NAME        0x00000005
*/
#define FILE_ACTION_MOVED                    0x00000006

class FileActionInfo {
public:
    LPWSTR    fileName;
    CHAR    drive;
    DWORD    action;
    time_t    timestamp;

    FileActionInfo(LPCWSTR fileName, CHAR drive, DWORD action) {
        this->fileName = (WCHAR*) GlobalAlloc(GPTR, sizeof(WCHAR)*(wcslen(fileName)+1));
        wcscpy(this->fileName, fileName);
        this->drive = drive;
        this->action = action;
        this->timestamp = time(NULL);
    }

    ~FileActionInfo() {
        GlobalFree(this->fileName);
    }
};

/*
There are two structures storing pointers to FileActionInfo items : a vector and a map.
This is because we need to be able to:
1) quickly retrieve the latest added item
2) quickly search among all queued items (in which case we use fileName as hashcode)
*/
class FileActionQueue {
private:
    vector<FileActionInfo*> *qActionQueue;
    map<wstring, vector<FileActionInfo*>> *mActionMap;

    void Queue(vector<FileActionInfo*> *v, FileActionInfo* lpAction) {
        v->push_back(lpAction);
    }

    void Dequeue(vector<FileActionInfo*> *v, FileActionInfo* lpAction) {
        for(int i = 0, nCount = v->size(); i < nCount; ++i){
            if(lpAction == v->at(i)) {
                v->erase(v->begin() + i);
                break;
            }
        }
    }

public:
    FileActionQueue() {
        this->qActionQueue = new vector<FileActionInfo*>;
        this->mActionMap = new map<wstring, vector<FileActionInfo*>>;
    }

    ~FileActionQueue() {
        delete qActionQueue;
        delete mActionMap;
    }

    void Add(FileActionInfo* lpAction) {
        this->Queue(&((*this->mActionMap)[lpAction->fileName]), lpAction);
        this->Queue(this->qActionQueue, lpAction);
    }

    void Remove(FileActionInfo* lpAction) {
        this->Dequeue(&((*this->mActionMap)[lpAction->fileName]), lpAction);
        this->Dequeue(this->qActionQueue, lpAction);
    }

    FileActionInfo* Last() {
        vector<FileActionInfo*> *v = this->qActionQueue;
        if(v->size() == 0) return NULL;
        return v->at(v->size()-1);
    }

    FileActionInfo* Search(LPCWSTR fileName, DWORD action, PCHAR drives) {
        FileActionInfo* result = NULL;
        vector<FileActionInfo*> *v;
        if( v = &((*this->mActionMap)[fileName])) {
            for(int i = 0, nCount = v->size(); i < nCount && !result; ++i){
                FileActionInfo* lpAction = v->at(i);
                if(wcscmp(lpAction->fileName, fileName) == 0 && lpAction->action == action) {
                    int j = 0;
                    while(drives[j] && !result) {
                        if(lpAction->drive == drives[j]) result = lpAction;
                        ++j;
                    }
                }
            }
        }
        return result;
    }
};
