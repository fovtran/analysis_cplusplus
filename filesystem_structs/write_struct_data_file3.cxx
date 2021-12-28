struct member_details
{
	float	member_length,																				//to store member length
			member_width,																				//to store member width
			member_depth,																				//to store member depth
			member_qk_factored_load,																	//to store unfactored load
			member_gk_factored_load,																	//to store factored load
			member_max_moment,																			//to store moment
			member_shear;																				//to store shear value
	int		member_type,																				//to store member type for quicker search
			member_no;																					//to display record number
	std::string	member_name;																			//to store member name e.g. beam or column
}record[100];

//load member details from file
void member_dimensions_file()
{
	 int	i,																						//declare and set int for loop
			no_of_members=100;																		//declare variable for while loop so that user enters a value less than 15
	string	filePath;																				//declare string for file name
	ifstream  dimensionsInFile;																		//declare input filestream name
	//cout<<"Enter full path name of file to save to"<<endl;										//Ask user to define file name to save to 
	//cin>> filePath;																				//user input for file path name
	dimensionsInFile.open ("test.dat");												
	if (!dimensionsInFile)
		{
			cout<<"Cannot load file"<<endl;
			return ;
		}
	else
		{
			for (i=0; i < no_of_members; i++)																//start of loop
				{	// write struct data from file	
					dimensionsInFile>>
					&record[i].member_depth,		
					&record[i].member_length,
					&record[i].member_width,
					&record[i].member_qk_factored_load,															
					&record[i].member_gk_factored_load,															
					&record[i].member_max_moment,																			
					&record[i].member_shear,																			
					&record[i].member_type,																				
					&record[i].member_no;	
					cout<<" Member no "<<i<<"stored"<<endl;
				}
			cout <<"All members have been successfully loaded"<<endl;
			dimensionsInFile.close();
		}
}

void member_details_save()
{
	 int	i,																						//declare and set int for loop
			no_of_members=100;																		//declare variable for while loop so that user enters a value less than 15
	string	filePath;																				//declare string for file name
	ofstream  dimensionsOutfile;																	//declare output filestream name
	//cout<<"Enter full path name of file to save to"<<endl;											//Ask user to define file name to save to 
	//cin>> filePath;																					//user input for file path name
	dimensionsOutfile.open ("test.dat");												
	if (!dimensionsOutfile)
		{
			cout<<"Cannot load file"<<endl;
			return ;
		}
	else
		{
			for (i=0; i < no_of_members; i++)																//start of loop
				{	// write struct data from file	
					dimensionsOutfile<<
					&record[i].member_depth,		
					&record[i].member_length,
					&record[i].member_width,
					&record[i].member_qk_factored_load,															
					&record[i].member_gk_factored_load,															
					&record[i].member_max_moment,																			
					&record[i].member_shear,																			
					&record[i].member_type,																				
					&record[i].member_no;	
					cout<<" Member no "<<i<<"stored"<<endl;
				}
			cout <<"All members have been successfully saved"<<endl;
			dimensionsOutfile.close();
		}
	
}

I don't think you need to reference that, it's enough with record[i].member_depth.

dimensionsOutfile<<
&record[i].member_depth,		
&record[i].member_length,
&record[i].member_width,
&record[i].member_qk_factored_load,	

dimensionsOutfile<<
record[i].member_depth <<	
record[i].member_length << ...

dimensionsOutfile<< ", " <<
record[i].member_depth << ", " <<
record[i].member_length << ...
