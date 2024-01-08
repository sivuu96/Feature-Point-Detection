/********************************************************************
Main Function for point cloud registration with Go-ICP Algorithm
Last modified: Feb 13, 2014

"Go-ICP: Solving 3D Registration Efficiently and Globally Optimally"
Jiaolong Yang, Hongdong Li, Yunde Jia
International Conference on Computer Vision (ICCV), 2013

Copyright (C) 2013 Jiaolong Yang (BIT and ANU)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*********************************************************************/

#include <time.h>
#include <iostream>
#include <fstream>
using namespace std;

#include "jly_goicp.h"
#include "ConfigMap.hpp"
#include ≠</opt/homebrew/bin/python3>
#include <Python.h>

#define DEFAULT_OUTPUT_FNAME "output.txt"
#define DEFAULT_CONFIG_FNAME "config.txt"
#define DEFAULT_MODEL_FNAME "model.txt"
#define DEFAULT_DATA_FNAME "data.txt"

void parseInput(int argc, char **argv, string &modelFName, string &dataFName, int &NdDownsampled, string &configFName, string &outputFName);
void readConfig(string FName, GoICP &goicp);
int loadPointCloud(string FName, int &N, POINT3D **p);

int main(int argc, char **argv)
{
	int i, Nm, Nd, Npoint, NdDownsampled;
	clock_t clockBegin, clockEnd;
	string modelFName, dataFName, pointdataFName, configFName, outputFname;
	POINT3D *pModel, *pData, *pointData, *temppointData;
	GoICP goicp;
	int threshold = 0.6;
	int decrementor=0.1;
	while (threshold > 0){
		// feature point calling comes here
		Py_Initialize();

		// Add the path of your script
		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append(\".\")"); // assuming the script is in the same directory

		// Import your Python script
		PyObject *pName = PyUnicode_DecodeFSDefault("featurePoints");
		PyObject *pModule = PyImport_Import(pName);
		Py_DECREF(pName);

		if (pModule != NULL){
			// Get the reference to your function
			PyObject *pFunc = PyObject_GetAttrString(pModule, "feature_point_calc");

			if (pFunc && PyCallable_Check(pFunc)){
				// Create a Python tuple to hold the arguments to the call
				PyObject *pArgs = PyTuple_New(1);
				PyTuple_SetItem(pArgs, 0, PyFloat_FromDouble(threshold)); // pass 0.01 as argument

				// Call the function
				PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
				Py_DECREF(pArgs);

				if (pValue != NULL){
					printf("Return of call : %ld\n", PyLong_AsLong(pValue));
					// pValue is the return value ie number of feature points
					Py_DECREF(pValue);
				}
			}
			else{
				if (PyErr_Occurred())
					PyErr_Print();
				fprintf(stderr, "Cannot find function \"feature_point_calc\"\n");
			}
			Py_XDECREF(pFunc);
			Py_DECREF(pModule);
		}

		else{
			PyErr_Print();
			fprintf(stderr, "Failed to load \"%s\"\n", "featurePoints");
			return 1;
		}
		
		Py_Finalize();


		parseInput(argc, argv, modelFName, dataFName, NdDownsampled, configFName, outputFname);
		readConfig(configFName, goicp);
		pointdataFName = argv[6];
		// Load model and data point clouds
		loadPointCloud(modelFName, Nm, &pModel);
		loadPointCloud(dataFName, Nd, &pData);
		loadPointCloud(pointdataFName, Npoint, &pointData);
		goicp.pModel = pModel;
		goicp.Nm = Nm;
		goicp.pData = pData;
		goicp.Nd = Nd;
		temppointData = (POINT3D *)malloc(sizeof(POINT3D) * Npoint);

		// Build Distance Transform
		cout << "Building Distance Transform..." << flush;
		clockBegin = clock();
		goicp.BuildDT();
		clockEnd = clock();
		cout << (double)(clockEnd - clockBegin) / CLOCKS_PER_SEC << "s (CPU)" << endl;

		// Run GO-ICP
		if (NdDownsampled > 0)
		{
			goicp.Nd = NdDownsampled; // Only use first NdDownsampled data points (assumes data points are randomly ordered)
		}
		cout << "Model ID: " << modelFName << " (" << goicp.Nm << "), Data ID: " << dataFName << " (" << goicp.Nd << ")" << endl;
		cout << "Registering..." << endl;
		clockBegin = clock();
		float squareError=goicp.Register();
		// save it to array
		//we are ending the loop where we get the rmse value

		// if rms<=0.01, break
		if(threshold==0.1)
		{decrementor=0.01;}
		else if (threshold==0)
		{//break not found
		}
		threshold-=decrementor;

		clockEnd = clock();
	}
	double time = (double)(clockEnd - clockBegin) / CLOCKS_PER_SEC;
	cout << "Optimal Rotation Matrix:" << endl;
	cout << goicp.optR << endl;
	cout << "Optimal Translation Vector:" << endl;
	cout << goicp.optT << endl;
	//	cout << "Finished in " << time << endl;

	ofstream ofile;
	ofile.open(outputFname.c_str(), ofstream::out);
	ofile << time << endl;
	ofile << goicp.optR << endl;
	ofile << goicp.optT << endl;
	ofile.close();
	for (i = 0; i < Npoint; i++)
	{
		POINT3D &p = pointData[i];
		temppointData[i].x = goicp.optR.val[0][0] * p.x + goicp.optR.val[0][1] * p.y + goicp.optR.val[0][2] * p.z + goicp.optT.val[0][0];
		temppointData[i].y = goicp.optR.val[1][0] * p.x + goicp.optR.val[1][1] * p.y + goicp.optR.val[1][2] * p.z + goicp.optT.val[1][0];
		temppointData[i].z = goicp.optR.val[2][0] * p.x + goicp.optR.val[2][1] * p.y + goicp.optR.val[2][2] * p.z + goicp.optT.val[2][0];
	}
	ofstream ofile1;
	clockEnd = clock();
	time = (double)(clockEnd - clockBegin) / CLOCKS_PER_SEC;
	cout << "Finished in " << time << endl;

	ofile1.open("budha_feature_reg12.txt", ofstream::out);
	for (i = 0; i < Npoint; i++)
	{
		ofile1 << temppointData[i].x << " " << temppointData[i].y << " " << temppointData[i].z << endl;
	}

	ofile1.close();

	delete (pModel);
	delete (pData);

	return 0;
}

void parseInput(int argc, char **argv, string &modelFName, string &dataFName, int &NdDownsampled, string &configFName, string &outputFName)
{ // Set default values
	modelFName = DEFAULT_MODEL_FNAME;
	dataFName = DEFAULT_DATA_FNAME;
	configFName = DEFAULT_CONFIG_FNAME;
	outputFName = DEFAULT_OUTPUT_FNAME;
	NdDownsampled = 0; // No downsampling

	// cout << endl;
	// cout << "USAGE:" << "./GOICP <MODEL FILENAME> <DATA FILENAME> <NUM DOWNSAMPLED DATA POINTS> <CONFIG FILENAME> <OUTPUT FILENAME>" << endl;
	// cout << endl;

	if (argc > 5)
	{
		outputFName = argv[5];
	}
	if (argc > 4)
	{
		configFName = argv[4];
	}
	if (argc > 3)
	{
		NdDownsampled = atoi(argv[3]);
	}
	if (argc > 2)
	{
		dataFName = argv[2];
	}
	if (argc > 1)
	{
		modelFName = argv[1];
	}

	cout << "INPUT:" << endl;
	cout << "(modelFName)->(" << modelFName << ")" << endl;
	cout << "(dataFName)->(" << dataFName << ")" << endl;
	cout << "(NdDownsampled)->(" << NdDownsampled << ")" << endl;
	cout << "(configFName)->(" << configFName << ")" << endl;
	cout << "(outputFName)->(" << outputFName << ")" << endl;
	cout << endl;
}

void readConfig(string FName, GoICP &goicp)
{
	// Open and parse the associated config file
	ConfigMap config(FName.c_str());

	goicp.MSEThresh = config.getF("MSEThresh");
	goicp.initNodeRot.a = config.getF("rotMinX");
	goicp.initNodeRot.b = config.getF("rotMinY");
	goicp.initNodeRot.c = config.getF("rotMinZ");
	goicp.initNodeRot.w = config.getF("rotWidth");
	goicp.initNodeTrans.x = config.getF("transMinX");
	goicp.initNodeTrans.y = config.getF("transMinY");
	goicp.initNodeTrans.z = config.getF("transMinZ");
	goicp.initNodeTrans.w = config.getF("transWidth");
	goicp.trimFraction = config.getF("trimFraction");
	// If < 0.1% trimming specified, do no trimming
	if (goicp.trimFraction < 0.001)
	{
		goicp.doTrim = false;
	}
	goicp.dt.SIZE = config.getI("distTransSize");
	goicp.dt.expandFactor = config.getF("distTransExpandFactor");

	cout << "CONFIG:" << endl;
	config.print();
	// cout << "(doTrim)->(" << goicp.doTrim << ")" << endl;
	cout << endl;
}

int loadPointCloud(string FName, int &N, POINT3D **p)
{
	int i;
	ifstream ifile;

	ifile.open(FName.c_str(), ifstream::in);
	if (!ifile.is_open())
	{
		cout << "Unable to open point file '" << FName << "'" << endl;
		exit(-1);
	}
	ifile >> N; // First line has number of points to follow
	*p = (POINT3D *)malloc(sizeof(POINT3D) * N);
	for (i = 0; i < N; i++)
	{
		ifile >> (*p)[i].x >> (*p)[i].y >> (*p)[i].z;
	}

	ifile.close();

	return 0;
}
