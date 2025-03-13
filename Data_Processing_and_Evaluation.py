import pandas as pd 
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pickle
import random
from collections import Counter
import copy
from transformers import BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

np.random.seed(42)
random.seed(42)


def get_opioid_list():

    """
    Reads an Excel file and extracts a list of opioid medications.

    Args:
        None

    Returns:
        list: A list of opioid medication names.
    """

	medications = pd.read_excel("medication_names_list.xlsx")
	opioids = medications["Opioids"].dropna().astype(str).str.strip().tolist()
	return opioids


def get_item_dict():

    """
    Reads ITEMID and LABEL from two CSV files (D_ITEMS.csv and D_LABITEMS.csv)
    and returns a dictionary mapping ITEMID to LABEL.

    Returns:
        dict: A dictionary where keys are ITEMIDs and values are LABELs.
    """

	file_paths = ["/project/boland_mimic_iii/D_ITEMS.csv","/project/boland_mimic_iii/D_LABITEMS.csv"]

	item_dict = {}
	for file in file_paths:
		data = pd.read_csv(file,low_memory=False)
		for index,row in data.iterrows():
			item_dict[row["ITEMID"]] = row["LABEL"]

	return item_dict


def get_prescription_df():

    """
    Extracts opioid prescription data from the PRESCRIPTIONS dataset.

    Args:
        None

    Returns:
        tuple: (opioid_list, filtered DataFrame)
            - opioid_list (str): Regex pattern of opioid names.
            - DataFrame: Filtered DataFrame containing opioid prescriptions.
    """

    # Load the opioid medication list
	opioid_list = get_opioid_list()

	opioid_list = "|".join(opioid_list)

	# Load the prescriptions data
	data = pd.read_csv("/project/boland_mimic_iii/PRESCRIPTIONS.csv", usecols=["HADM_ID", "STARTDATE", "ENDDATE", "DRUG"], low_memory=False)

	# Filter only opioid prescriptions
	data = data[data["DRUG"].str.contains(opioid_list,case=False)]

	# Rename columns and add metadata 
	data = data.rename(columns={"STARTDATE":"STARTTIME","ENDDATE":"ENDTIME","DRUG":"LABEL"})
	data["ITEMID"] = -1
	data["LABEL"] = "OPIOID PRESCRIPTION"

	# Convert dates to datetime format
	data['STARTTIME']= pd.to_datetime(data['STARTTIME'])
	data['ENDTIME'] = pd.to_datetime(data['ENDTIME'])

	return opioid_list, data



def get_static_data():

    """
    Extracts static patient data from ADMISSIONS and PATIENTS datasets.

    Args:
		None

    Returns:
        tuple: (static_data, static_data_column_index)
            - static_data (dict): Patient static feature vectors indexed by HADM_ID.
            - static_data_column_index (dict): Mapping of column indices to feature names.
    """

    # Load admissions data
	admissions = pd.read_csv("/project/boland_mimic_iii/ADMISSIONS.csv",low_memory=False)
	admissions = admissions[["SUBJECT_ID","HADM_ID","ADMISSION_TYPE","ADMISSION_LOCATION","INSURANCE","RELIGION","MARITAL_STATUS","ETHNICITY"]]

	# Load patient data
	patients = pd.read_csv("/project/boland_mimic_iii/PATIENTS.csv")
	patients = patients[["SUBJECT_ID","GENDER"]]

	# Merge admissions with patient gender
	all_data = admissions.join(patients.set_index("SUBJECT_ID"), on="SUBJECT_ID")

	# Obtain age groups
	age_group = obtain_patient_age_groups()
	age_group_str = ["0-18","18-25","25-35","35-50","50-65","65+"]
	all_data["Age"] = all_data["HADM_ID"].apply(lambda x: age_group_str[age_group[x]])

	# Set index to HADM_ID
	all_data = all_data.set_index("HADM_ID")
 
 	# One-hot encode categorical variables
	all_data = pd.get_dummies(all_data).drop(columns=["SUBJECT_ID"])

	# Create index mapping
	columns = all_data.columns
	static_data_column_index = {i:a for i,a in enumerate(columns)}

	# Convert to dictionary format
	static_data = {}
	for index,row in all_data.iterrows():
		static_data[index] = row.values

	return static_data, static_data_column_index


def get_procedure_events_df():

    """
    Extracts procedure events from MIMIC-III, maps ITEMID to human-readable labels.

    Args:
        None.

    Returns:
        pd.DataFrame: Processed procedure events with columns [HADM_ID, STARTTIME, ENDTIME, ITEMID, LABEL].
    """

    # Load item dictionary
	item_dict = get_item_dict()

	# Load procedure events dataset
	data = pd.read_csv("/project/boland_mimic_iii/PROCEDUREEVENTS_MV.csv",low_memory=False)

	# Convert timestamps to datetime format
	data['STARTTIME']= pd.to_datetime(data['STARTTIME'])
	data['ENDTIME'] = pd.to_datetime(data['ENDTIME'])

	# Keep only necessary columns
	data = data[["HADM_ID","STARTTIME","ENDTIME","ITEMID"]]

	# Map ITEMID to its label
	data["LABEL"] = data["ITEMID"].apply(lambda x: item_dict[x])

	print(datetime.datetime.now(), ": Finished Processing Procedure Events")

	return data

def get_input_or_lab_or_output_events(data_type = "input",mv = True):

    """
    Retrieves input, lab, or output events from MIMIC-III dataset.

    Args:
        data_type (str, optional): One of ["input", "lab", "output"]. Defaults to "input".
        mv (bool, optional): If True, loads the Metavision (MV) dataset; otherwise, uses CareVue (CV) for inputs. Defaults to True.

    Returns:
        pd.DataFrame: Processed events with columns [HADM_ID, STARTTIME, ENDTIME, ITEMID, LABEL].
    """

	if data_type == "input" and mv:
		filename = "/project/boland_mimic_iii/INPUTEVENTS_MV.csv"
		extension = "Input Events MV"
	elif data_type == "input" and not mv:
		filename = "/project/boland_mimic_iii/INPUTEVENTS_CV.csv"
		extension = "Input Events CV"
	elif data_type == "lab":
		filename = "/project/boland_mimic_iii/LABEVENTS.csv"
		extension = "LAB Events"
	elif data_type == "output":
		filename = "/project/boland_mimic_iii/OUTPUTEVENTS.csv"
		extension = "OUTPUT Events"

	item_dict = get_item_dict()

	data = pd.read_csv(filename,low_memory=False)   

	if data_type == "input" and mv:
		data = data[["HADM_ID","STARTTIME","ENDTIME","ITEMID"]]
	else:
		data = data[["HADM_ID","CHARTTIME","ITEMID"]]
		data = data.rename(columns={"CHARTTIME":"STARTTIME"})
		data["ENDTIME"] = data["STARTTIME"]

	data['STARTTIME']= pd.to_datetime(data['STARTTIME'])
	data['ENDTIME'] = pd.to_datetime(data['ENDTIME'])
	data["LABEL"] = data["ITEMID"].apply(lambda x: item_dict[x])

	print(datetime.datetime.now(), ": Finished Processing ",extension)

	return data


def get_combined_data():

    """
    Combines procedure events, input events (MV & CV), and output events into a single DataFrame.

    Returns:
        pd.DataFrame: A sorted DataFrame containing all events, indexed by hospital admission (HADM_ID).
    """

	procedure_events = get_procedure_events_df()
	input_events_mv = get_input_or_lab_or_output_events("input",True)
	input_events_cv = get_input_or_lab_or_output_events("input",False)
	output_events = get_input_or_lab_or_output_events("output",False)

	data = pd.concat([procedure_events,input_events_mv,input_events_cv,output_events]) 
	del procedure_events
	del input_events_mv
	del input_events_cv 
	del output_events 

	data = data.sort_values(by=['HADM_ID','STARTTIME'],ascending = (True, True))
	data = data.reset_index(drop=True)

	return data



def prepare_training_data(): 

    """
    Prepares training data by extracting event sequences and static data,
    then saves them for classification tasks.

    Returns:
    		None
    """

    # Load event and static data
	events = get_combined_data()
	static_data, static_data_column_index = get_static_data() 

	save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/Dependence/" 

	# Save static data column index
	with open(save_path + "static_data_column_index","wb") as f:
		pickle.dump(static_data_column_index,f) 

 
	# Load positive opioid dependence IDs
	with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Opioid_Dependence/Positive_Ids","rb") as f:
		pos_hadmid = pickle.load(f) 

	# Get negative IDs (those not in pos_hadmid)
	neg_hadmid = list(set(events["HADM_ID"]) - pos_hadmid)

	# Create labels
	labels = {a:1 for a in pos_hadmid}
	labels.update({a:0 for a in neg_hadmid})


	# Prepare event sequences and static data
	hadmid_events_pos = {}
	hadmid_events_neg = {}
	static_data_pos = {}
	static_data_neg = {}

	for hadmid in labels.keys():
		temp_data = events[events["HADM_ID"]==hadmid]
		event_sequence = []
		for index,row in temp_data.iterrows():
			event_sequence.append(row['ITEMID'])

		
		if labels[hadmid] == 0:
			if len(event_sequence) > 0:
				hadmid_events_neg[hadmid] = event_sequence
			if str(hadmid) != "nan":
				static_data_neg[hadmid] = static_data[hadmid]
		elif labels[hadmid] == 1:
			if len(event_sequence) > 0:
				hadmid_events_pos[hadmid] = event_sequence
			if str(hadmid) != "nan":
				static_data_pos[hadmid] =static_data[hadmid] 

 
	# Create item ID index mapping
	item_id_index = {a:i for i,a in enumerate(set([item for sublist in [hadmid_events_pos[a] for a in hadmid_events_pos.keys()] + [hadmid_events_neg[a] for a in hadmid_events_neg.keys()] for item in sublist]))}

	with open(save_path + "item_id_index","wb") as f:
		pickle.dump(item_id_index,f)


	with open(save_path + "hadmid_events","wb") as f:
		pickle.dump((hadmid_events_pos,hadmid_events_neg),f)


	with open(save_path + "hadmid_static","wb") as f:  
		pickle.dump((static_data_pos, static_data_neg),f) 
 
 
def get_note_events():

	"""
    Loads and processes note events from MIMIC-III dataset.

    Returns a DataFrame with selected columns.
    """

	data = pd.read_csv("/project/boland_mimic_iii/NOTEEVENTS.csv",low_memory=False)
	data = data[["HADM_ID","CHARTDATE","TEXT","ROW_ID","CATEGORY"]]
	data = data.rename(columns={"CHARTDATE":"STARTTIME"})
	data["Priority"] = 1
	return data


def get_positive_negative_examples():
 
    """
    This function extracts and processes clinical notes to create positive and negative training examples
    based on opioid dependence labels. It then saves the processed data.

    Returns:
        hadmid_notes_pos (dict): Dictionary containing positive examples (opioid dependence) with hospital admission IDs as keys.
        hadmid_notes_neg (dict): Dictionary containing negative examples (no opioid dependence) with hospital admission IDs as keys.
    """

	print("Getting Training Examples")

	start = datetime.datetime.now()

	# Load clinical notes
	notes = get_note_events()

	print("Finished Loading Notes: ",datetime.datetime.now() - start)

	# Load positive hospital admission IDs (HADM_IDs) from a pre-saved file
	with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Opioid_Dependence/Positive_Ids","rb") as f:
		pos_hadmid = pickle.load(f) 

	# Determine negative HADM_IDs (those not in positive set)
	neg_hadmid = set(notes['HADM_ID']) - pos_hadmid

	# Create a dictionary of labels (1 for positive, 0 for negative)
	labels = {a:1 for a in pos_hadmid}
	labels.update({a:0 for a in neg_hadmid})

	# Sort notes by HADM_ID, STARTTIME, and Priority
	data = notes.sort_values(by=['HADM_ID','STARTTIME','Priority'],ascending = (True, True, True))
	data = data.reset_index(drop=True)

	print("Finished Sorting Data: ",datetime.datetime.now() - start)

	# Get unique hospital admission IDs
	all_hadmid = list(set(data["HADM_ID"]))


	hadmid_notes_pos = {}
	hadmid_notes_neg = {}

	# Iterate through each hospital admission ID
	for hadmid in tqdm(all_hadmid):

		# Filter data for the current HADM_ID
		temp_data = data[data["HADM_ID"] == hadmid] 

		notes_list = []

		# Process each row in the filtered data
		for index,row in temp_data.iterrows():
			if row["CATEGORY"] != "Discharge summary":
				notes_list.append({"TEXT":row["TEXT"],"ROW_ID":row["ROW_ID"],"CATEGORY":row["CATEGORY"],"CHARTDATE":row["STARTTIME"]})

		# Store notes based on label (positive or negative)
		if len(notes_list) > 0:
			if labels[hadmid] == 1:
				hadmid_notes_pos[hadmid] = notes_list[:]
			elif labels[hadmid] == 0:
				hadmid_notes_neg[hadmid] = notes_list[:]

	save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/Dependence/" 

	with open(save_path + "Text_Hadmid","wb") as f:
		pickle.dump((hadmid_notes_pos,hadmid_notes_neg),f)

	return hadmid_notes_pos, hadmid_notes_neg  



def undersample_newborn_neg_data(pos_ids,all_data,datatype):

    """
    This function performs undersampling on negative examples related to newborn admissions.
    It selects a balanced subset of positive and negative cases based on the "ADMISSION_TYPE_NEWBORN" attribute.

    Args:
        pos_ids (set): Set of positive example IDs.
        all_data (dict): Dictionary containing all data with keys as patient IDs.
        datatype (str): Type of data being processed.

    Returns:
        all_ids (list): List of balanced positive and negative IDs.
        extra_neg_minus (list): List of extra negative examples not included in the balanced set.
    """

	path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/"
	with open(path + datatype + "/static_data_column_index",'rb') as f:
		mapping = pickle.load(f)

	# Identify the index corresponding to "ADMISSION_TYPE_NEWBORN"
	considered_index = [a for a in mapping.keys() if mapping[a] == "ADMISSION_TYPE_NEWBORN"][0]

	pos = 0
	neg = 0
	tot_pos = 0
	tot_neg = 0

	pos_plus = []
	pos_minus = []
	neg_plus = []
	neg_minus = []

	# Iterate through all patient records
	for key in all_data.keys():
		if key in pos_ids:
			pos += all_data[key]["static"][considered_index]
			tot_pos += 1

			if all_data[key]["static"][considered_index] == 1:
				pos_plus.append(key)
			else:
				pos_minus.append(key)

		else:
			neg += all_data[key]["static"][considered_index]
			tot_neg += 1

			if all_data[key]["static"][considered_index] == 1:
				neg_plus.append(key)
			else:
				neg_minus.append(key)
		
	# Determine the minimum count between positive and negative non-newborn cases
	num_minus = np.min([len(pos_minus),len(neg_minus)])

	# Select extra negative cases beyond the balanced subset (up to 5000 additional)
	extra_neg_minus = neg_minus[num_minus:5000+num_minus] 

	# Balance the positive and negative samples
	pos_minus, neg_minus = pos_minus[:num_minus], neg_minus[:num_minus]

	# Combine selected positive and negative IDs and shuffle
	all_ids = pos_minus + neg_minus
	random.shuffle(all_ids)
	
	return all_ids, extra_neg_minus


def create_train_test_data():

    """
    This function processes structured and unstructured clinical data to create training, validation,
    and test datasets for machine learning models. It tokenizes text data using ClinicalBERT,
    balances the dataset by undersampling newborn negative cases, and saves the processed data.
    """

	test_size = 0.05 # Percentage of data used for testing
	val_size = 0.05 # Percentage of data used for validation

	path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/"
	datatypes = ["Dependence","Prescription"]

	# Load ClinicalBERT tokenizer
	tokenizer = BertTokenizer.from_pretrained("/home/kashyap/ClinicalBERT/biobert_pretrain_output_all_notes_150000")


	hadmids_cat = {}
	all_data = {}
	balanced_ids = {}
	extra_neg_ids = {} 

	for datatype in datatypes:

		# Load item ID index
		with open(path + datatype + "/item_id_index","rb") as f:
			item_id_index = pickle.load(f)

		hadmids_cat[datatype] = {}
		all_data[datatype] = {}

		# Load structured time-varying event data
		with open(path + datatype + "/hadmid_events","rb") as f:
			(ev_pos,ev_neg) = pickle.load(f)
		ev_data = copy.deepcopy(ev_pos)
		ev_data.update(ev_neg)
		ev_data = {int(a):ev_data[a] for a in ev_data.keys()}

		# Load structured static data 
		with open(path + datatype + "/hadmid_static","rb") as f:
			(static_pos, static_neg) = pickle.load(f)
		static_data = copy.deepcopy(static_pos)
		static_data.update(static_neg)
		static_data = {int(a):static_data[a] for a in static_data.keys()}

		# Load clinical notes text data
		with open(path + datatype + "/Text_Hadmid","rb") as f:
			(text_pos, text_neg) = pickle.load(f)
		text_data = copy.deepcopy(text_pos)
		text_data.update(text_neg)
		text_data = {int(a):text_data[a] for a in text_data.keys()} 

		# Categorize HADM_IDs into positive and negative groups
		hadmids_cat[datatype]["pos"] = set(ev_pos.keys()) | set(static_pos.keys()) | set(text_pos.keys())
		hadmids_cat[datatype]["neg"] = set(ev_neg.keys()) | set(static_neg.keys()) | set(text_neg.keys())


		all_data[datatype] = {}
		error_1 = 0
		error_2 = 0
		error_3 = 0
		total = 0

		# Process and structure all data
		for hadmid in hadmids_cat[datatype]["pos"] | hadmids_cat[datatype]["neg"]:
			hadmid = int(hadmid)

			all_data[datatype][hadmid] = {}

			# Add structured event data
			if hadmid in ev_data.keys():
				all_data[datatype][hadmid]["events"] = [0] + [item_id_index[a] + 2 for a in ev_data[hadmid]] + [1]
			else:
				all_data[datatype][hadmid]["events"] = [0,1]
				error_1 += 1

			# Add static data
			if hadmid in static_data.keys():
				all_data[datatype][hadmid]["static"] = static_data[hadmid]
			else:
				all_data[datatype][hadmid]["static"] = [0 for a in range(105)]
				error_2 += 1

			# Add text data
			if hadmid in text_data.keys():
				all_data[datatype][hadmid]["text"] = [a['TEXT'] for a in text_data[hadmid]]
			else:
				all_data[datatype][hadmid]["text"] = ["No Data Available"]
				error_3 += 1

			total += 1

		print(datatype,total - error_1, total - error_2,total - error_3)


		# Perform undersampling to balance dataset
		balanced_ids[datatype], extra_neg_ids[datatype] = undersample_newborn_neg_data(hadmids_cat[datatype]["pos"],all_data[datatype],datatype)

		# Split data into Train, Validation, and Test sets
		ind_beg = {}
		ind_end = {}
		ind_beg["Train"] = 0
		ind_end["Train"] = int((1-val_size-test_size) * len(balanced_ids[datatype]))
		ind_beg["Val"] = int((1-val_size-test_size) * len(balanced_ids[datatype]))
		ind_end["Val"] = int((1-test_size) * len(balanced_ids[datatype]))
		ind_beg["Test"] = int((1-test_size) * len(balanced_ids[datatype]))
		ind_end["Test"] = len(balanced_ids[datatype])


		for split in ["Train","Val","Test"]:

			X_static = []
			X_structured = []
			X_text = []
			y = []

			considered_hadmids = balanced_ids[datatype][ind_beg[split]:ind_end[split]]
			if split == "Test":
				considered_hadmids = considered_hadmids + extra_neg_ids[datatype]

			for hadmid in considered_hadmids:
				X_static.append(all_data[datatype][hadmid]["static"])
				X_structured.append(all_data[datatype][hadmid]["events"])
				X_text.append(all_data[datatype][hadmid]["text"])
				label = (1 if hadmid in hadmids_cat[datatype]["pos"] else 0) 
				y.append(label)

			# Tokenize text data using ClinicalBERT
			X_text = [[tokenizer._tokenize(a) for a in b] for b in tqdm(X_text)]
			X_text = [[[tokenizer._convert_token_to_id(a) for a in b] for b in c] for c in tqdm(X_text)]
			X_text = [[tokenizer.build_inputs_with_special_tokens(a) for a in b] for b in tqdm(X_text)]

			save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/" + split + "_Data_1/"

			with open(save_path + datatype,"wb") as f:
				pickle.dump((X_static,X_structured,X_text,y),f)
		

def correct_static_data():

    """
    Processes and corrects static patient data by:
    - Removing unwanted static features based on predefined ignored categories.
    - Ensuring consistency across training, validation, and test datasets.
    - Adding an "UNKNOWN" category for categorical features where applicable.
    """

    # Load ignored categories from a text file
	with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/Ignored_Static_Data.txt","r") as f:
		ignored_categories = set(f.read().split("\n"))

	save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/"

	for datatype in ["Prescription","Dependence"]: 

		base_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/" + datatype + "/static_data_column_index"

		# Load static data column mapping
		with open(base_path,"rb") as f:
			data_dict = pickle.load(f)

		# Extend ignored categories with additional unwanted features
		ignored_categories = ignored_categories | set([data_dict[a] for a in data_dict.keys() if "DISCHARGE" in data_dict[a]])
		ignored_categories = ignored_categories | set(["RELIGION_NOT SPECIFIED","RELIGION_UNOBTAINABLE",'ETHNICITY_UNABLE TO OBTAIN',"ETHNICITY_UNKNOWN/NOT SPECIFIED"])

		# Filter out ignored columns from static data
		data_dict = {a:data_dict[a] for a in data_dict.keys() if data_dict[a] not in ignored_categories} 

		considered_inds = [a for a in data_dict.keys()]
		considered_columns = [data_dict[a] for a in data_dict.keys()]

		# Process training, validation, and test datasets
		for split in ["Train_Data","Val_Data","Test_Data"]:
			new_dict = {i:a for i,a in enumerate(considered_columns)}

			with open(save_path + split + "_1/" + datatype,"rb") as f:
				(X_static,X_structured,X_text,y) = pickle.load(f)

				# Retain only the considered columns
				X_static = [[a[b] for b in considered_inds] for a in X_static]
				X_static = np.array(X_static)

				# Add "UNKNOWN" category for categorical features
				for feature_name in ["ADMISSION_TYPE","ADMISSION_LOCATION","INSURANCE","RELIGION","MARITAL_STATUS","ETHNICITY","GENDER","Age"]:
					feature_inds = [a for a in new_dict.keys() if feature_name in new_dict[a]]
					X_static_temp = np.sum(X_static[:,feature_inds[0]:feature_inds[-1] + 1],axis=1)
					unknown_col = np.ones((len(X_static_temp),)) - X_static_temp
					unknown_col = unknown_col.reshape((len(unknown_col),1))
					X_static = np.concatenate((X_static,unknown_col),axis=1)
					new_dict[len(new_dict)] = feature_name + "_UNKNOWN"

			# Save the corrected dataset
			with open(save_path + split + "_1/" + datatype + "_corrected","wb") as f:
				pickle.dump((X_static,X_structured,X_text,y),f) 

		# Save updated feature mapping
		with open(base_path + "_corrected","wb") as f:
			pickle.dump(new_dict,f)  

			
def get_top_features():

    """
    Identifies the top positive and negative features from structured, static, or text data using logistic regression.
    The mode determines the type of data to analyze ("static", "structured", or "text").
    """

	mode = "text" # Choose from "static", "structured" or "text"

	save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/"

	for datatype in ["Dependence","Prescription"]:
		print(datatype)
		all_X_static = []
		all_X_structured = []
		all_X_text = []
		all_y = []

		# Load data from training, validation, and test sets
		for split in ["Train_Data","Val_Data","Test_Data"]:
			with open(save_path + split + "/" + datatype + "_corrected","rb") as f:
				(X_static,X_structured,X_text,y) = pickle.load(f)
				all_X_static += [a for a in X_static]
				all_X_structured += X_structured
				all_X_text += X_text
				all_y += y


		if mode == "structured":
			# Load feature dictionary
			with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/" + datatype + "/item_id_index","rb") as f:
				data_dict = pickle.load(f)
				data_dict = {data_dict[a]:a for a in data_dict.keys()}

			# Convert structured data into binary feature vectors
			for i in tqdm(range(len(all_X_structured))):
				all_X_structured[i] = [a - 2 for a in all_X_structured[i][1:-1]]
				temp = np.zeros(len(data_dict))
				for j in range(len(all_X_structured[i])):
					temp[all_X_structured[i][j]] = 1
				all_X_structured[i] = temp

		elif mode == "text":
			# Prepare text data for vectorization
			all_X_text = [" ".join(a[-40:]) for a in all_X_text]
			cv = CountVectorizer(stop_words='english',ngram_range=(1,3),max_features=20000,binary=True)
			all_X_text = cv.fit_transform(all_X_text)
	

		# Split into training and test sets
		np.random.seed(42)
		indices = np.random.choice(len(all_X_static),len(all_X_static),replace=False)
		indices_train = indices[:int(0.9 * len(indices))]
		indices_test = indices[int(0.9 * len(indices)):]

		X_static_train = [all_X_static[a] for a in indices_train]
		X_structured_train = [all_X_structured[a] for a in indices_train]
		X_text_train = [all_X_text[a] for a in indices_train]
		y_train = [all_y[a] for a in indices_train]

		X_static_test = [all_X_static[a] for a in indices_test]
		X_structured_test = [all_X_structured[a] for a in indices_test]
		X_text_test = [all_X_text[a] for a in indices_test]
		y_test = [all_y[a] for a in indices_test]


		clf = LogisticRegression()

		if mode == "static":
			clf.fit(X_static_train,y_train)
			y_pred = clf.predict(X_static_test)
			print(datatype,mode,accuracy_score(y_pred,y_test),"Num Features",len(X_static_train[0]))

			base_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/" + datatype + "/static_data_column_index_corrected"

			# Load feature names
			with open(base_path,"rb") as f:
				data_dict = pickle.load(f)

			# Identify top positive and negative features
			pos_features = np.argsort(-clf.coef_[0])[:20]
			print("\nPOS FEATURES\n")
			for feature in pos_features:
				print(data_dict[feature])


			print("\n\n\nNEG FEATURES\n") 
			neg_features = np.argsort(clf.coef_[0])[:20]
			for feature in neg_features:
				print(data_dict[feature])


		elif mode == "structured":
			clf.fit(X_structured_train,y_train)
			y_pred = clf.predict(X_structured_test)

			print(datatype,mode,accuracy_score(y_pred,y_test))

			# Load mapping of item IDs to labels
			id_name_mapping = pd.read_csv("/project/boland_mimic_iii/D_ITEMS.csv",low_memory = False)
			id_name_mapping = {row["ITEMID"]:row["LABEL"] for index,row in id_name_mapping.iterrows()}

			pos_features = np.argsort(-clf.coef_[0])[:20]
			print("\nPOS FEATURES\n")
			for feature in pos_features:
				print(id_name_mapping[data_dict[feature]])


			print("\n\n\nNEG FEATURES\n") 
			neg_features = np.argsort(clf.coef_[0])[:20]
			for feature in neg_features:
				print(id_name_mapping[data_dict[feature]])



		elif mode == "text":
			clf.fit(X_text_train,y_train)
			y_pred = clf.predict(X_text_test)

			print(datatype,mode,accuracy_score(y_pred,y_test))

			# Extract vocabulary mapping
			vocab = cv.vocabulary_
			vocab = {vocab[a]:a for a in vocab.keys()}
 
			pos_features = np.argsort(-clf.coef_[0])[:20]
			print("\nPOS FEATURES\n")
			for feature in pos_features:
				print(vocab[feature])


			print("\n\n\nNEG FEATURES\n") 
			neg_features = np.argsort(clf.coef_[0])[:20]
			for feature in neg_features:
				print(vocab[feature]) 




def obtain_patient_age_groups():

   """
    This function processes patient age data and categorizes patients into age groups.
    - If mode is "create": It loads patient and admission data, calculates ages, assigns age groups,
      and saves the mapping of hospital admission IDs (HADM_ID) to age groups.
    - If mode is "load": It loads and returns the saved age group data.
    
    Returns:
        dict: A dictionary mapping HADM_ID to an age group if mode is "load".
    """

	mode = "load" # Choose from either "create" or "load"

	def obtain_age(x):

	"""Calculates the patient's age at the time of admission."""


		try:
			return (x["ADMITTIME"] - x["DOB"]).days // 365
		except:
			return -1 # Return -1 if there's an error (e.g., missing data).

	def obtain_age_group(x):

		"""Assigns an age group based on the patient's age."""

		age = x["Age"]

		if age < 0:
			return age # Keep invalid ages (-1) as they are.
		elif age >= 0 and age < 18: # Pediatric (0-17)
			return 0
		elif age >= 18 and age < 25: # Young adult (18-24)
			return 1
		elif age >= 25 and age < 35: # Adult (25-34)
			return 2
		elif age >= 35 and age < 50: # Middle-aged adult (35-49)
			return 3
		elif age >= 50 and age < 65: # Older adult (50-64)
			return 4
		elif age >= 65: # Senior (65+)
			return 5

	if mode == "create":

		# Load patient demographic data and extract relevant columns.
		patients = pd.read_csv("/project/boland_mimic_iii/PATIENTS.csv")
		patients = patients[["SUBJECT_ID","DOB"]]
		patients = patients.set_index("SUBJECT_ID")

		# Load admissions data and extract relevant columns.
		admissions = pd.read_csv("/project/boland_mimic_iii/ADMISSIONS.csv")
		admissions = admissions[["SUBJECT_ID","HADM_ID","ADMITTIME"]]
		admissions = admissions.set_index("SUBJECT_ID")

		# Merge patient and admission data based on SUBJECT_ID.
		final_data = patients.join(admissions)

		# Convert date columns to datetime objects.
		for column in ["DOB","ADMITTIME"]:
			final_data[column] =  pd.to_datetime(final_data[column]).dt.date

		# Compute patient ages and assign age groups.
		final_data["Age"] = final_data.apply(obtain_age,axis=1)
		final_data["Age_Group"] = final_data.apply(obtain_age_group,axis=1)

		# Create a dictionary mapping HADM_ID to Age_Group.
		final_dict = {row["HADM_ID"]:row["Age_Group"] for index,row in final_data.iterrows()}

		with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/HADM_ID-Age","wb") as f:
			pickle.dump(final_dict,f)

	elif mode == "load":

		with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/HADM_ID-Age","rb") as f:
			data = pickle.load(f)

		return data





if __name__ == "__main__":


	start = datetime.now()

	prepare_training_data()
	create_train_test_data() 
	

	end = datetime.now()

	print("Total Run Time: ",end-start)



