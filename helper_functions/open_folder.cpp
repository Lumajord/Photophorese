
#include "open_folder.h"

namespace myfoldercreator
{

void result_folder::open_file(std::ofstream &file, const std::ostringstream &filename)
{
	this->files.push_back(&file);			// save file
	file.open((this->path + filename.str()).c_str());	// open file
	if(!file)
	{
		std::cout << "Error: couldn't open file" << std::endl << std::endl;
	}
	else
	{
		std::cout << "opened File: " << (this->path + filename.str()) << std::endl << std::endl;
	}
}





void result_folder::open_file(std::ofstream &file, const std::string &filename)
{
    this->files.push_back(&file);			// save file
    file.open((this->path + filename).c_str());	// open file
    if(!file)
    {
        std::cout << "Error: couldn't open file" << std::endl << std::endl;
    }
    else
    {
        std::cout << "opened File: " << (this->path + filename) << std::endl << std::endl;
    }
}





void result_folder::open_file(std::ofstream &file, const char* filename)
{
    this->files.push_back(&file);			// save file
    file.open((this->path + std::string(filename)).c_str());	// open file
    if(!file)
    {
        std::cout << "Error: couldn't open file" << std::endl << std::endl;
    }
    else
    {
        std::cout << "opened File: " << (this->path + std::string(filename)) << std::endl << std::endl;
    }
}





void result_folder::open_file(std::ofstream &file)
{
    this->files.push_back(&file);			// save file
    std::string filename;
    std::ostringstream osst;
    osst << "unnamedfile" << this->files.size();
    filename = osst.str();

    file.open((this->path + filename).c_str());	// open file
    if(!file)
    {
        std::cout << "Error: couldn't open file" << std::endl << std::endl;
    }
    else
    {
        std::cout << "opened File: " << (this->path + filename) << std::endl << std::endl;
    }
}





result_folder::~result_folder(void)
{

    // close files
	for(unsigned int i = 0; i < this->files.size(); ++i)
		this->files[i]->close();
}

void result_folder::close_all()
{
    // close files
	for(unsigned int i = 0; i < this->files.size(); ++i)
		this->files[i]->close();
}






result_folder::result_folder(const std::string &folder_name, int up)
{


        char basePath[256];

        // The manpage says it won't null terminate.  Let's zero the buffer.
        memset(basePath, 0, sizeof(basePath));

// get path to the programm
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

        if (GetModuleFileNameA( NULL, basePath, sizeof(basePath)/sizeof(basePath[0]) ) < 0) {
            perror("readlink");
        }

#else


        // Note we use sizeof(buf)-1 since we may need an extra char for NUL.
        if (readlink("/proc/self/exe", basePath, sizeof(basePath)-1) < 0)
        {
           // There was an error...  Perhaps the path does not exist
           // or the buffer is not big enough.  errno has the details.
           perror("readlink");
           //return -1;
        }


#endif

        // cut off the path at the parent folder up 'up' times from the programm
        for(int i = sizeof(basePath)/sizeof(basePath[0])-1; i > -1; --i)
        {
            if(basePath[i] == slash)
                up--;

            if(up == 0)
                basePath[i] = '\0';

        }

        this->basepath = std::string(basePath);
        this->path = std::string(basePath) + folder_name + slash;

// create folder
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

        if(CreateDirectoryA(this->path.c_str(), NULL)) //  || ERROR_ALREADY_EXISTS == GetLastError()
            std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;


#else

        struct stat st;
        if (stat(this->path.c_str(), &st) == -1)
        {
            mkdir(this->path.c_str(), 0700);
            std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;
        }


#endif

}





result_folder::result_folder(const std::ostringstream &folder_name, int up)
{


        char basePath[256];

        // The manpage says it won't null terminate.  Let's zero the buffer.
        memset(basePath, 0, sizeof(basePath));

// get path to the programm
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

        if (GetModuleFileNameA( NULL, basePath, sizeof(basePath)/sizeof(basePath[0]) ) < 0) {
            perror("readlink");
        }

#else


        // Note we use sizeof(buf)-1 since we may need an extra char for NUL.
        if (readlink("/proc/self/exe", basePath, sizeof(basePath)-1) < 0)
        {
           // There was an error...  Perhaps the path does not exist
           // or the buffer is not big enough.  errno has the details.
           perror("readlink");
           //return -1;
        }


#endif

        // cut off the path at the parent folder up 'up' times from the programm
        for(int i = sizeof(basePath)/sizeof(basePath[0])-1; i > -1; --i)
        {
            if(basePath[i] == slash)
                up--;

            if(up == 0)
                basePath[i] = '\0';

        }

        this->basepath = std::string(basePath);
        this->path = std::string(basePath) + folder_name.str() + slash;

// create folder
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

        if(CreateDirectoryA(this->path.c_str(), NULL)) //  || ERROR_ALREADY_EXISTS == GetLastError()
            std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;


#else

        struct stat st;
        if (stat(this->path.c_str(), &st) == -1)
        {
            mkdir(this->path.c_str(), 0700);
            std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;
        }


#endif

}





result_folder::result_folder(const char* folder_name, int up)
{


        char basePath[256];

        // The manpage says it won't null terminate.  Let's zero the buffer.
        memset(basePath, 0, sizeof(basePath));

// get path to the programm
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

        if (GetModuleFileNameA( NULL, basePath, sizeof(basePath)/sizeof(basePath[0]) ) < 0) {
            perror("readlink");
        }

#else


        // Note we use sizeof(buf)-1 since we may need an extra char for NUL.
        if (readlink("/proc/self/exe", basePath, sizeof(basePath)-1) < 0)
        {
           // There was an error...  Perhaps the path does not exist
           // or the buffer is not big enough.  errno has the details.
           perror("readlink");
           //return -1;
        }


#endif

        // cut off the path at the parent folder up 'up' times from the programm
        for(int i = sizeof(basePath)/sizeof(basePath[0])-1; i > -1; --i)
        {
            if(basePath[i] == slash)
                up--;

            if(up == 0)
                basePath[i] = '\0';

        }

        this->basepath = std::string(basePath);
        this->path = std::string(basePath) + std::string(folder_name) + slash;

// create folder
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

        if(CreateDirectoryA(this->path.c_str(), NULL)) //  || ERROR_ALREADY_EXISTS == GetLastError()
            std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;


#else

        struct stat st;
        if (stat(this->path.c_str(), &st) == -1)
        {
            mkdir(this->path.c_str(), 0700);
            std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;
        }


#endif

}




result_folder::result_folder(int up)
{


        char basePath[256];

        // The manpage says it won't null terminate.  Let's zero the buffer.
        memset(basePath, 0, sizeof(basePath));

// get path to the programm
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

        if (GetModuleFileNameA( NULL, basePath, sizeof(basePath)/sizeof(basePath[0]) ) < 0) {
            perror("readlink");
        }

#else


        // Note we use sizeof(buf)-1 since we may need an extra char for NUL.
        if (readlink("/proc/self/exe", basePath, sizeof(basePath)-1) < 0)
        {
           // There was an error...  Perhaps the path does not exist
           // or the buffer is not big enough.  errno has the details.
           perror("readlink");
           //return -1;
        }


#endif

        // cut off the path at the parent folder up 'up' times from the programm
        for(int i = sizeof(basePath)/sizeof(basePath[0])-1; i > -1; --i)
        {
            if(basePath[i] == slash)
                up--;

            if(up == 0)
                basePath[i] = '\0';

        }

        this->basepath = std::string(basePath);

}




void result_folder::create_folder(const std::string &folder_name)
{

    this->path = this->basepath + folder_name + slash;

    // create folder
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

    if(CreateDirectoryA(this->path.c_str(), NULL)) //  || ERROR_ALREADY_EXISTS == GetLastError()
        std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;


    #else

    struct stat st;
    if (stat(this->path.c_str(), &st) == -1)
    {
        mkdir(this->path.c_str(), 0700);
        std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;
    }


    #endif

}

void result_folder::create_folder(const char* folder_name)
{

    this->path = this->basepath + std::string(folder_name) + slash;

    // create folder
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

    if(CreateDirectoryA(this->path.c_str(), NULL)) //  || ERROR_ALREADY_EXISTS == GetLastError()
        std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;


    #else

    struct stat st;
    if (stat(this->path.c_str(), &st) == -1)
    {
        mkdir(this->path.c_str(), 0700);
        std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;
    }


    #endif

}





void result_folder::create_folder(const std::ostringstream &folder_name)
{

    this->path = this->basepath + folder_name.str() + slash;

    // create folder
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

    if(CreateDirectoryA(this->path.c_str(), NULL)) //  || ERROR_ALREADY_EXISTS == GetLastError()
        std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;


    #else

    struct stat st;
    if (stat(this->path.c_str(), &st) == -1)
    {
        mkdir(this->path.c_str(), 0700);
        std::cout << std::endl << "Succesfully created folder: " << this->path << std::endl << std::endl;
    }


    #endif

}





} // namespace myfoldercreator
