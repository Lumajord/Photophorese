/*
 * Copyright (c) 2016 L. Jordan
 *
 * This class creates a folder in a parent directory of the running programm
 * and allows for easy opening of files for Windows and Linux operating systems.
 * Files are always opened in the last folder that was created.
 * All opened files will be closed automatically.
*/
#ifndef OPEN_FOLDER_H
#define OPEN_FOLDER_H


#include <fstream> //includes for writing to file
#include <sstream>
#include <string>

#include<vector>
#include<iostream>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

static const char slash='\\';
#include <windows.h>


#else

static const char slash='/';
#include <sys/types.h>
#include <sys/stat.h>
#include<cstring>
#include<unistd.h>

#endif

namespace myfoldercreator
{


class result_folder
{
public:
	
    /*
     * open_file:
     * opens and saves a file
     *
     * @param file      : File to be opened.
     * @param filename  : Name of the file to be opened
     *                    file is opened in the last folder that was created
    */
	void open_file(std::ofstream &file, const std::ostringstream &filename);
    void open_file(std::ofstream &file, const std::string &filename);
    void open_file(std::ofstream &file, const char* filename);
    void open_file(std::ofstream &file);


    /*
     * close_all:
     * closes all opened files
    */
    void close_all();

    /*
     * constructor:
     * creates a folder in a parent folder of the running programm
     *
     * @param folder_name   : Name of the folder that will be created
     * @param up            : The folder will be created in the parent directory
     *                        'up' times up from the running programm
    */
    explicit result_folder(const std::string &folder_name, const int up = 0);
    explicit result_folder(const char* folder_name, const int up = 0);
    explicit result_folder(const std::ostringstream &folder_name, const int up = 0);
    explicit result_folder(const int up = 0);


    /*
     * create folder:
     * creates a folder inside basepath
     *
     * @param folder_name   : Name of the folder that will be created
    */
    void create_folder(const std::string &folder_name);
    void create_folder(const std::ostringstream &folder_name);
    void create_folder(const char* folder_name);



    /*
     * destructor:
     * closes all open files
    */
	~result_folder(void);
	
private:
    /*
     * files:
     * containes the pointers to all the opened files so that they can be closed later
    */
	std::vector<std::ofstream*> files;

    /*
     * path:
     * path to the folder
    */
	std::string path;

    /*
     * basepath:
     * path to the folder's parent folder
    */
    std::string basepath;

};

} // namespace myfoldercreator


#endif // OPEN_FOLDER_H
