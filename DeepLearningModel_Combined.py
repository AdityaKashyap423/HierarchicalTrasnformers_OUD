import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from torch.autograd import Variable
import random
from transformers import BertConfig, BertTokenizer, BertModel, BertForPreTraining
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.calibration import IsotonicRegression




class Event_Sequence_and_Static_Model(nn.Module):

    """
    A neural network model for processing event sequences with optional static features. 
    The model supports both LSTM and Transformer architectures.

    Attributes:
        model_type (str): Specifies the type of sequence model ('lstm' or 'transformer').
        use_static_features (bool): Indicates whether static features are included in the input.
        num_static_features (int): Number of static features used if enabled.
        embedding_size (int): Size of the embedding vectors for event sequences.
        sequence_length (int): Length of the input event sequence.
        lstm_hidden_size (int): Number of hidden units in the LSTM layers.
        num_lstm_layers (int): Number of LSTM layers.
        embedding_layer (nn.Embedding): Embedding layer for event sequences.
        embedding_layer_position (nn.Embedding): Positional embedding for transformer model.
        lstm (nn.LSTM): LSTM network for sequence modeling.
        transformer (nn.Transformer): Transformer model for sequence modeling.
        ffl_lstm (nn.Linear): Feedforward layer for LSTM output processing.
        ffl_transformer (nn.Linear): Feedforward layer for Transformer output processing.
    """


	def __init__(self, event_sequence_model, number_of_events, sequence_length, use_static_features=False, num_static_features=0):
		
        """
        Initializes the event sequence model.

        Args:
            event_sequence_model (str): 'lstm' or 'transformer' to specify the model type.
            number_of_events (int): Number of unique event types in the sequence.
            sequence_length (int): Length of the event sequence input.
            use_static_features (bool, optional): Whether to include static features. Defaults to False.
            num_static_features (int, optional): Number of static features if enabled. Defaults to 0.
        """

		super(Event_Sequence_and_Static_Model, self).__init__()

		self.model_type = event_sequence_model
		self.use_static_features = use_static_features
		self.num_static_features = num_static_features
		if not self.use_static_features:
			self.num_static_features = 0
		self.embedding_size = 40
		self.sequence_length = sequence_length
		self.lstm_hidden_size = 20
		self.num_lstm_layers = 4

		# Event embedding layer with padding 
		self.embedding_layer = nn.Embedding(number_of_events + 91, self.embedding_size, padding_idx=0)

		# Positional embeddings for transformer model
		self.embedding_layer_position = nn.Embedding(sequence_length, self.embedding_size)

		# LSTM network: bidirectional with multiple layers
		self.lstm = nn.LSTM(self.embedding_size, self.lstm_hidden_size, self.num_lstm_layers, batch_first = True, bidirectional=True)

		# Transformer model with encoder-decoder architecture
		self.transformer = torch.nn.Transformer(d_model=self.embedding_size, nhead=4, num_encoder_layers=2, num_decoder_layers=2,dim_feedforward=1024)

		# Fully connected layers for classification
		self.ffl_lstm = nn.Linear(self.sequence_length * self.lstm_hidden_size * 2 + self.num_static_features, 50)
		self.ffl_transformer = nn.Linear(self.embedding_size + self.num_static_features, 50)

	def forward(self, x, x_static):

        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, sequence_length).
            x_static (torch.Tensor): Static feature tensor of shape (batch_size, num_static_features).

        Returns:
            torch.Tensor: Output of shape (batch_size, 50), representing structured static and time varying event embeddings.
        """

        # Embed input time-varying event sequences
		x = self.embedding_layer(x)

		# Process using LSTM model
		if self.model_type == 'lstm':

			# Initialize hidden and cell states for LSTM (num_layers * 2 for bidirectional)
			h0 = Variable(torch.zeros(self.num_lstm_layers * 2, x.shape[0],self.lstm_hidden_size))
			c0 = Variable(torch.zeros(self.num_lstm_layers * 2, x.shape[0],self.lstm_hidden_size))

			if torch.cuda.is_available():
				h0 = h0.cuda()
				c0 = c0.cuda()

			# Forward pass through LSTM
			x, (hn, cn) = self.lstm(x,(h0,c0))

		# Process using Transformer model
		elif self.model_type == 'transformer':

			# Create position indices for positional embeddings
			pos_range = torch.tensor(list(range(x.shape[1])))

			if torch.cuda.is_available():
				pos_range = pos_range.cuda()

			# Expand positional embeddings to match input size
			x_pos = self.embedding_layer_position(pos_range)
			x_pos = torch.transpose(x_pos.expand(x.size()),0,1)

			# Create padding mask (True for padded elements)
			padding_mask = torch.sum(x,2) == 0
			if torch.cuda.is_available():
				padding_mask = padding_mask.cuda()

			# Transformer requires (seq_len, batch, embedding_dim) format
			x = torch.transpose(x,0,1)

			# Apply Transformer model with positional encoding and convert back to (batch, seq_len, embedding_dim) format
			x = torch.transpose(self.transformer(src = x + x_pos,tgt = x + x_pos,src_key_padding_mask = padding_mask, tgt_key_padding_mask = padding_mask),0,1)

		# Process LSTM output
		if self.model_type == 'lstm':

			# Flatten sequence output
			x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])

			# Concatenate static features if enabled
			if self.use_static_features:
				x = torch.cat((x,x_static),1)

			# Apply fully connected layer to obtain static and time-varying event embeddings 
			x = self.ffl_lstm(x)

		# Process Transformer output
		elif self.model_type == 'transformer':

			# Extract representation from the first token
			x = x[:,0,:]

			# Concatenate static features if enabled
			if self.use_static_features:
				x = torch.cat((x,x_static),1)

			# Apply fully connected layer to obtain static and time-varying event embeddings
			x = self.ffl_transformer(x)

		return x





class Text_Model(nn.Module):

    """
    A hierarchical transformer-based model that processes clinical notes.
    
    - Extracts embeddings using BioBERT.
    - Uses transformers for intra-note and inter-note attention.
    - Predicts outcomes based on processed patient notes.
    """

	def __init__(self, number_of_notes_per_patient, number_of_note_splices):

		super(Text_Model, self).__init__()

		# Define embedding sizes
		self.BertEmbeddingSize = 768 # BERT output embedding size
		self.reducedEmbeddingSize = 32 # Reduced embedding size after projection

		# Store input parameters
		self.number_of_notes_per_patient = number_of_notes_per_patient
		self.number_of_note_splices = number_of_note_splices

		# Load BioBERT model
		self.BERTconfig = BertConfig.from_json_file("/home/kashyap/ClinicalBERT/biobert_pretrain_output_all_notes_150000/config.json")
		self.BERTmodel = BertModel.from_pretrained("/home/kashyap/ClinicalBERT/biobert_pretrain_output_all_notes_150000/pytorch_model.bin", config = self.BERTconfig)

		# Linear layer to reduce BERT embedding size
		self.ffl_1 = nn.Linear(self.BertEmbeddingSize, self.reducedEmbeddingSize)

		# Two Transformer layers for intra-note and inter-note attention
		self.first_transformer = torch.nn.Transformer(d_model=self.reducedEmbeddingSize,nhead=4,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=512)
		self.second_transformer = torch.nn.Transformer(d_model=self.reducedEmbeddingSize,nhead=4,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=512)

		# Positional Embeddings
		self.embedding_layer_1 = torch.nn.Embedding(num_embeddings=number_of_note_splices, embedding_dim=self.reducedEmbeddingSize)
		self.embedding_layer_2 = torch.nn.Embedding(num_embeddings=number_of_notes_per_patient, embedding_dim=self.reducedEmbeddingSize)

		# Final linear layer for prediction
		self.final_ffl = nn.Linear(self.reducedEmbeddingSize,50) 

	def forward(self, x):

        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, notes, splices, token_dim=512)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 50)
        """

        # Reshape input tensor (Flatten batch and notes dimensions)
		x_shape = x.size() # Expected: (batch_size=32, notes=40, splices=16, token_dim=512)
		x = x.view(-1,x_shape[2],x_shape[3]) # (notes * batch_size, splices, 512)

		# Track note indices: Keeping track of sentence numbers
		note_list = torch.Tensor(list(range(x_shape[0])))
		note_list = torch.transpose(note_list.expand((x_shape[1],x_shape[0])),0,1)
		note_list = torch.reshape(note_list,(x_shape[1]*x_shape[0],1))

		# Filter out empty inputs
		considered_notes = x.sum(dim=(1,2)) != 0
		note_list = note_list[considered_notes].squeeze()
		x = x[considered_notes]

		# Reshape for processing splices
		x_shape_interim = x.shape
		x = x.view(-1, x_shape[3])  # (total_splices, 512)

		# Track splice indices
		splices_list = torch.Tensor(list(range(x_shape_interim[0])))
		splices_list = torch.transpose(splices_list.expand((x_shape_interim[1],x_shape_interim[0])),0,1)
		splices_list = torch.reshape(splices_list,(x_shape_interim[1]*x_shape_interim[0],1))


		# Filter out empty splices
		considered_splices = x.sum(dim=(1)) != 0
		splices_list = splices_list[considered_splices].squeeze()	
		x = x[considered_splices]

		# Move to GPU if available
		if torch.cuda.is_available():
			x = x.cuda()

		# Freeze BERT parameters (No fine-tuning)
		for param in self.BERTmodel.parameters():
			param.requires_grad = False

		# Get BERT embeddings
		bert_pad = x != 0
		bert_pad = bert_pad.type(torch.LongTensor)
		x = self.BERTmodel(x,encoder_attention_mask = bert_pad)[1]
		x = self.ffl_1(x)	


		# Reconstruct note-splice structure
		splices_list = splices_list.tolist() 
		if type(splices_list) == float:
			splices_list = [splices_list]
		splices_list_counter = Counter(splices_list)
		splices_list_counter = [splices_list_counter[a] for a in range(len(splices_list_counter))]
		x = torch.split(x,splices_list_counter)
		x = pad_sequence(x, batch_first=True, padding_value=0)

	# First Transformer: Intra note-splice attention

		# Position embeddings
		pos_range_1 = torch.tensor(list(range(x.shape[1])))
		if torch.cuda.is_available():
			pos_range_1 = pos_range_1.cuda()
		x_pos_1 = self.embedding_layer_1(pos_range_1)
		x_pos_1 = torch.transpose(x_pos_1.expand(x.size()),0,1)

		# Create padding mask
		padding_mask_1 = torch.sum(x,2) == 0
		if torch.cuda.is_available():
			padding_mask_1 = padding_mask_1.cuda()
		x = torch.transpose(x,0,1) + x_pos_1
		x = torch.transpose(self.first_transformer(src = x,tgt = x,src_key_padding_mask = padding_mask_1, tgt_key_padding_mask = padding_mask_1),0,1)
		x = x[:,0,:]


		# Reconstructing batch-note distribution
		note_list = note_list.tolist()
		if type(note_list) == float:
			note_list = [note_list]

		# Reconstruct batch-note structure
		note_list_counter = Counter(note_list)
		note_list_counter = [note_list_counter[a] for a in range(len(note_list_counter))]
		x = torch.split(x,note_list_counter)
		x = pad_sequence(x, batch_first=True, padding_value=0)

	# Second Transformer: Inter patient-note attention

		# Position Embeddings
		pos_range_2 = torch.tensor(list(range(x.shape[1])))
		if torch.cuda.is_available():
			pos_range_2 = pos_range_2.cuda()
		x_pos_2 = self.embedding_layer_2(pos_range_2)
		x_pos_2 = torch.transpose(x_pos_2.expand(x.size()),0,1)

		# Padding
		padding_mask_2 = torch.sum(x,2) == 0
		if torch.cuda.is_available():
			padding_mask_2 = padding_mask_2.cuda()
		x = torch.transpose(x,0,1) + x_pos_2
		x = torch.transpose(self.second_transformer(src = x,tgt = x,src_key_padding_mask = padding_mask_2, tgt_key_padding_mask = padding_mask_2),0,1)
		x = x[:,0,:]

		# Final outcome prediction
		x = self.final_ffl(x)

		return x



class Combined_Model(nn.Module):

    """
    A multimodal deep learning model that integrates structured (event-based and static features)
    and unstructured (clinical notes) data from patient EHRs for clinical prediction.

    Attributes:
        Structured_Model (Event_Sequence_and_Static_Model): Handles structured and event sequence data.
        Unstructured_Model (Text_Model): Handles unstructured text data using BERT and Transformers.
        ffl (nn.Linear): A final fully connected layer mapping combined features to output.
        activation (nn.Softmax): Applies softmax for classification.
    """

	def __init__(self, event_sequence_model, number_of_events, sequence_length, number_of_notes_per_patient, number_of_note_splices, use_static_features=False, num_static_features=0):

        """
        Initializes the combined model with structured and unstructured data processing pipelines.

        Args:
            event_sequence_model (nn.Module): Predefined event sequence model.
            number_of_events (int): Number of unique event types.
            sequence_length (int): Maximum length of event sequences.
            number_of_notes_per_patient (int): Maximum number of notes per patient.
            number_of_note_splices (int): Number of text slices per note.
            use_static_features (bool, optional): Whether to use static features. Defaults to False.
            num_static_features (int, optional): Number of static features if used. Defaults to 0.
        """

		super(Combined_Model, self).__init__()

		# Model for structured static and event-based features
		self.Structured_Model = Event_Sequence_and_Static_Model(event_sequence_model, number_of_events, sequence_length, use_static_features, num_static_features)

		# Model for unstructured text data
		self.Unstructured_Model = Text_Model(number_of_notes_per_patient, number_of_note_splices)

        # Fully connected layer to combine structured and unstructured representations
		self.ffl = nn.Linear(100,2)  # 50 from each model → output size 2 (for classification)

		# Softmax activation for classification
		self.activation = nn.Softmax(dim=1) 


	def forward(self, x, x_static, x_text):

        """
        Forward pass of the combined model.

        Args:
            x (torch.Tensor): Input tensor for structured event sequence data.
            x_static (torch.Tensor): Input tensor for static features.
            x_text (torch.Tensor): Input tensor for unstructured clinical text.

        Returns:
            torch.Tensor: Probability distribution over classes.
        """

        # Process structured event sequence and static features
		x_structured = self.Structured_Model(x,x_static)

		# Process unstructured clinical text
		x_unstructured = self.Unstructured_Model(x_text)

		# Concatenate both representations
		x = torch.cat([x_structured,x_unstructured],axis=1) 

		# Final classification output
		x = self.activation(self.ffl(x))
		return x 


def load_data_2(task, datatype, max_seq_len):

    """
    Loads structured, static, and unstructured text data for a given task and data split.

    Args:
        task (str): Task name (e.g., "Prescription").
        datatype (str): Data split type ("train", "val", or "test").
        max_seq_len (int): Maximum sequence length for structured data.

    Returns:
        tuple: (X_structured, X_static, X_text, y)
            - X_structured (torch.Tensor): Padded structured event sequences.
            - X_static (list): Static features.
            - X_text (list): Unstructured clinical notes.
            - y (list): Labels.
    """

    # Define the base directory for data storage
	save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/"

	# Mapping datatype to corresponding directory
	datatype_map = {}
	datatype_map["train"] = "Train_Data_1/"
	datatype_map["val"] = "Val_Data_1/"
	datatype_map["test"] = "Test_Data_1/"

	# Load the data from a pickle file
	with open(save_path + datatype_map[datatype] + task + "_corrected","rb") as f:
		(X_static,X_structured,X_text,y) = pickle.load(f)

	# Truncate structured sequences to max_seq_len and convert to tensors
	X_structured = [a[-max_seq_len:] for a in X_structured] # Keep only the last max_seq_len elements

	# Convert to PyTorch tensors
	X_structured = [torch.Tensor(a) for a in X_structured]

	# Pad sequences with -1 (later shifted to 0 by adding 1)
	X_structured = pad_sequence(X_structured, batch_first = True, padding_value=-1) + 1

	# Subsample training and validation data for the "Prescription" task. "Dependence" dataset is smaller and doesn't require subsampling.
	if datatype == "train" and task == "Prescription":
		X_static,X_structured,X_text,y = X_static[:3000],X_structured[:3000],X_text[:3000],y[:3000]

	if datatype == "val" and task == "Prescription":
		X_static,X_structured,X_text,y = X_static[:400],X_structured[:400],X_text[:400],y[:400]

	return (X_structured,X_static,X_text,y)





class MyDataSet(Dataset):

    """
    Custom PyTorch Dataset for handling structured, sequence, and unstructured text data.

    Args:
        datatype (str): Data split type ("train", "val", or "test").
        max_notes_per_patient (int): Maximum number of notes per patient.
        max_sequences_per_note (int): Maximum number of note splices per note.
        task (str): Task name (e.g., "Prescription").
        max_structured_sequence_length (int): Maximum sequence length for structured data.

    Attributes:
        X_structured (torch.Tensor): Structured data sequences.
        X_static (list): Static features.
        X_text (list): Unstructured text (clinical notes).
        y (list): Labels.
        tokenizer (BertTokenizer): Tokenizer for handling clinical text.
        pad_id (int): Padding token ID used for text inputs.
    """

	def __init__(self,datatype, max_notes_per_patient, max_sequences_per_note,task, max_structured_sequence_length):
		super(MyDataSet, self).__init__()
		self.task = task
		self.max_structured_sequence_length = max_structured_sequence_length
		self.datatype = datatype
		self.bert_input_length = 512 # Maximum token length per BERT input
		self.max_notes_per_patient = max_notes_per_patient
		self.max_sequences_per_note = max_sequences_per_note

		# Load dataset using `load_data_2`
		self.X_structured, self.X_static, self.X_text, self.y = load_data_2(self.task, self.datatype, self.max_structured_sequence_length) 

		# Initialize BERT tokenizer
		self.tokenizer = BertTokenizer.from_pretrained("/home/kashyap/ClinicalBERT/biobert_pretrain_output_all_notes_150000")
		self.pad_id = self.tokenizer._convert_token_to_id("[PAD]")

	def __len__(self):

		"""Returns the total number of samples in the dataset."""
		return len(self.y)

	def __getitem__(self,index):

        """
        Retrieves a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (X_structured, X_static, X_text, y)
                - X_structured (torch.Tensor): Structured sequence input data.
                - X_static (torch.Tensor): Structured static input features.
                - X_text (torch.Tensor): Tokenized and padded text input.
                - y (int/float): Target label.
        """

		X_structured = self.X_structured[index]
		X_static = torch.Tensor(self.X_static[index])
		current_instance = self.X_text[index]
		y = self.y[index]

		X_text = []

		# Process each clinical note in the patient record
		for note in current_instance:
			X_note = []

			# Determine the number of splits required per note (each split fits into BERT's 512 token limit)
			num_parts_per_note = int(len(note)/self.bert_input_length) + 1
			for i in range(num_parts_per_note):

				# Extract the i-th slice of the note
				current_splice = note[i * self.bert_input_length:(i + 1) * self.bert_input_length]

				# Pad the last slice if it's shorter than the expected length
				if len(current_splice) < self.bert_input_length: # PADDING PER NOTE SPLICE
					current_splice = current_splice + [self.pad_id] * (self.bert_input_length - len(current_splice))

				X_note.append(current_splice)

			# Pad or truncate the note-level sequences to `max_sequences_per_note`
			if len(X_note) < self.max_sequences_per_note:

				# Pad with empty slices if there are fewer than max_sequences_per_note
				X_note += [[self.pad_id for a in range(self.bert_input_length)] for a in range(self.max_sequences_per_note - len(X_note))]

			elif len(X_note) > self.max_sequences_per_note:

				# Truncate to keep only the first `max_sequences_per_note`
				X_note = X_note[:self.max_sequences_per_note]

			X_text.append(X_note)

		# Pad or truncate patient-level notes to `max_notes_per_patient`
		if len(X_text) < self.max_notes_per_patient:

			# Pad with empty notes if there are fewer than max_notes_per_patient
			X_text += [[[self.pad_id for a in range(self.bert_input_length)] for a in range(self.max_sequences_per_note)] for a in range(self.max_notes_per_patient - len(X_text))]

		elif len(X_text) > self.max_notes_per_patient:

			# Truncate to keep only the last `max_notes_per_patient`
			X_text = X_text[-self.max_notes_per_patient:]

		# Convert to PyTorch tensor
		X_text = torch.Tensor(X_text)

		return (X_structured, X_static, X_text, y)

	def data_info(self):

	    """
	    Retrieves metadata about the dataset, including the number of event types 
	    and the size of static features.

	    Returns:
	        tuple:
	            num_events (int): Total number of unique event types.
	            static_size (int): Number of static features.
	    """

	    # Define the dataset directory
		save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/"

		# Mapping data type to folder name
		datatype_map = {}
		datatype_map["train"] = "Train_Data/"
		datatype_map["val"] = "Val_Data/"
		datatype_map["test"] = "Test_Data/"


		# Load structured static and time-varying dataset
		with open(save_path + datatype_map[self.datatype] + self.task + "_corrected","rb") as f:
			(X_static,X_structured,X_text,y) = pickle.load(f)

		# Extract static feature size
		static_size = len(X_static[0])

		# Load event ID index file
		with open(save_path + self.task + "/item_id_index","rb") as f:
			data = pickle.load(f)

		# Compute number of unique event types (+3 for special cases)
		num_events = len(data) + 3

		return num_events, static_size



def get_evaluation(y_true, y_prob):

    """
    Computes accuracy score between true labels and predicted labels.

    Args:
        y_true (Tensor or ndarray): Ground truth labels.
        y_prob (Tensor or ndarray): Model's predicted probability scores.

    Returns:
        float: Accuracy score.
    """

	# Convert predicted probabilities to class labels using argmax
	try:
		y_prob = np.argmax(y_prob.cpu(), axis=1)
	except:
		y_prob = np.argmax(y_prob, axis=1)

	# Compute accuracy
	try:
		return accuracy_score(y_true.cpu(), y_prob)
	except:
		return accuracy_score(y_true, y_prob)


 

def train():


    """
    Train or test a transformer-based model for clinical event classification.
    The function loads datasets, initializes models, and optimizes them using SGD.
    
    Modes:
        - "train": Trains the model and saves the best version.
        - "test": Loads a saved model and evaluates it on the test set.
    """

	mode = "test" # Set mode to either 'train' or 'test'
	task = "Dependence" # Set task to either 'Prescription' or 'Dependence'

	# Hyperparameters
	num_epoches = 5
	batch_size = 4
	learning_rate = 0.001
	momentum = 0.9
	use_static_features = True
	event_sequence_model = 'transformer'
	number_of_notes_per_patient = 40 
	number_of_note_splices = 10
	sequence_length = 200


	for seed in range(10): # Iterate over different random seeds

		best_loss = float('inf') # Track best validation loss
		best_epoch = 0


		save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/saved_models/Final_Models" 
	 
		# Define dataset parameters
		training_params = {"batch_size": batch_size, "shuffle": True, "drop_last": False}
		test_params = {"batch_size": batch_size, "shuffle": False, "drop_last": False}

		# Load datasets
		training_set = MyDataSet("train",number_of_notes_per_patient,number_of_note_splices,task,sequence_length)
		training_generator = DataLoader(training_set, **training_params)
		val_set = MyDataSet("val",number_of_notes_per_patient,number_of_note_splices,task,sequence_length)
		val_generator = DataLoader(val_set, **test_params)
		test_set = MyDataSet("val",number_of_notes_per_patient,number_of_note_splices,task,sequence_length)
		test_generator = DataLoader(test_set, **test_params)

		# Extract dataset info
		number_of_events, num_static_features = training_set.data_info() 
	 

		print("NUMBER OF EVENTS: ",number_of_events)
		print("MAX SEQUENCE LENGTH: ", sequence_length)
		print("NUM STATIC FEATURES: ", num_static_features)

		# Define loss function
		criterion = nn.CrossEntropyLoss()
	 
		if mode == "train":

			# Initialize model
			model = Combined_Model(event_sequence_model, number_of_events, sequence_length, number_of_notes_per_patient, number_of_note_splices, use_static_features=use_static_features, num_static_features=num_static_features)

			if torch.cuda.is_available():
				model = model.cuda()
				model = nn.DataParallel(model) # Parallelize computation
		 
			optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum)

			num_iter_per_epoch = len(training_generator)
			eval_every = int(num_iter_per_epoch/2) # Evaluate halfway through each epoch

			training_curve_metrics = []

			# Training loop
			for epoch in range(num_epoches):
				for iter, (feature_structured, feature_static, feature_text, label) in enumerate(training_generator):

					feature_structured = feature_structured.type(torch.LongTensor)
					feature_text = feature_text.type(torch.LongTensor)

					if torch.cuda.is_available():
						feature_structured = feature_structured.cuda()
						feature_static = feature_static.cuda()
						label = label.cuda()

					optimizer.zero_grad()
		 
					try:

						predictions = model(feature_structured, feature_static, feature_text)

						loss = criterion(predictions, label)
						loss.backward()
						optimizer.step()

						with torch.no_grad():
							print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
								epoch + 1,
								num_epoches,
								iter + 1,
								num_iter_per_epoch,
								learning_rate,
								loss,
								get_evaluation(label,predictions)))  

					except RuntimeError as e:  
						print("Batch didn't fit in memory")
						for p in model.parameters():
							if p.grad is not None:
								del p.grad  # free some memory
						torch.cuda.empty_cache()

					# Periodic validation
					if iter % eval_every == 0:
						model.eval()
						loss_ls = []
						te_label_ls = []
						te_pred_ls = []
						for te_feature_structured, te_feature_static, te_feature_text, te_label in tqdm(val_generator):
							num_sample = len(te_label)
							te_feature_structured = te_feature_structured.type(torch.LongTensor)
							te_feature_text = te_feature_text.type(torch.LongTensor)
							if torch.cuda.is_available():
								te_feature_structured = te_feature_structured.cuda()
								te_feature_static = te_feature_static.cuda()
								te_label = te_label.cuda()
							try:
								with torch.no_grad():
									te_predictions = model(te_feature_structured, te_feature_static, te_feature_text)
								te_loss = criterion(te_predictions, te_label)
								loss_ls.append(te_loss * num_sample)
								te_label_ls.extend(te_label.clone().cpu())
								te_pred_ls.append(te_predictions.clone().cpu()) 
							except RuntimeError as e:
									print("Batch didn't fit in memory")
									for p in model.parameters():
										if p.grad is not None:
											del p.grad  # free some memory
									torch.cuda.empty_cache()

						te_loss = sum(loss_ls) / test_set.__len__()
						te_pred = torch.cat(te_pred_ls, 0)
						te_label = np.array(te_label_ls)
						test_metrics = get_evaluation(te_label, te_pred.numpy())
						model.train()

						print("Validation Set Accuracy: ",test_metrics,' Loss: ',te_loss)
						training_curve_metrics.append((len(training_curve_metrics), test_metrics))

						plot_training_curve(seed, task, training_curve_metrics)  

				# Save best model
				if te_loss < best_loss:
					best_loss = te_loss
					best_epoch = epoch
					torch.save(model, save_path + os.sep + task + "_Transformer_Combined_Classifier_Final_" + str(seed))

		elif mode == "test":

			# Load trained model
			model = torch.load(save_path + os.sep + task + "_Transformer_Combined_Classifier_Final_" + str(seed)) 
 
			model.eval()

			loss_ls = []
			te_label_ls = []
			te_pred_ls = []

			for te_feature_structured, te_feature_static, te_feature_text, te_label in tqdm(test_generator):
				num_sample = len(te_label)
				te_feature_structured = te_feature_structured.type(torch.LongTensor)
				te_feature_text = te_feature_text.type(torch.LongTensor)

				if torch.cuda.is_available():
					te_feature_structured = te_feature_structured.cuda()
					te_feature_static = te_feature_static.cuda()
					te_label = te_label.cuda()

				try:
					with torch.no_grad():
						te_predictions = model(te_feature_structured, te_feature_static, te_feature_text)
					te_loss = criterion(te_predictions, te_label)
					loss_ls.append(te_loss * num_sample)
					te_label_ls.extend(te_label.clone().cpu())
					te_pred_ls.append(te_predictions.clone().cpu())
				except RuntimeError as e:
					print("Batch didn't fit in memory")
					for p in model.parameters():
						if p.grad is not None:
							del p.grad  # free some memory
					torch.cuda.empty_cache()

			te_loss = sum(loss_ls) / test_set.__len__()
			te_pred = torch.cat(te_pred_ls, 0)
			te_label = np.array(te_label_ls)
			test_metrics = get_evaluation(te_label, te_pred.numpy())

			print("TEST Set Accuracy: ",test_metrics,' Loss: ',te_loss)  

			with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/Predictions/" + task + "_proba_final_val_" + str(seed),"wb") as f:
				pickle.dump((te_pred.cpu(),te_label),f)    

			del model 


def plot_training_curve(seed, task, training_curve_metrics):

    """
    Plots and saves the training curve showing validation accuracy over epochs.

    Parameters:
    seed (int): The random seed used for the training run.
    task (str): The classification task (e.g., "Dependence" or "Prescription").
    training_curve_metrics (list of tuples): A list of (epoch, validation accuracy) tuples.

    Returns:
    None
    """

	plt.plot([a[0] for a in training_curve_metrics],[a[1] for a in training_curve_metrics])
	plt.xlabel("Epochs")
	plt.ylabel("Val Accuracy")
	plt.title("Training Curve")

	plt.savefig("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Training_Curves/" + task + "_" + str(seed) + ".png")  




def get_evaluation_metrics():

    """
    Computes and prints evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC, and specificity)
    for multiple random seeds on two classification tasks: 'Prescription' and 'Dependence'.

    Returns:
    None
    """

	base_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/Predictions/"

	datatypes = ["Prescription","Dependence"]

	for datatype in datatypes:

		acc_list = []
		prec_list = []
		recall_list = []
		f1_list = []
		roc_auc_score_list = []
		specificity_list = []

		seed_list = [0,1,2,3,4,5,6,7,8,9]
			
		for seed in seed_list:
			with open(base_path + datatype + "_proba_final_" + str(seed),"rb") as f:
				(pred_proba,test) = pickle.load(f)

				#-------------------------- FOR RANDOM PERFORMANCE --------------------------#
				# Uncomment below to test with random predictions instead of model predictions
				# pred_proba = np.random.rand(pred_proba.shape[0],pred_proba.shape[1])
				#----------------------------------------------------------------------------#

				# Convert prediction probabilities to class predictions
				pred = np.argmax(np.array(pred_proba.tolist()),axis=1)
				test = np.array(list(test))

				acc_list.append(accuracy_score(pred,test))
				metrics = precision_recall_fscore_support(test, pred, average='binary')
				prec_list.append(metrics[0])
				recall_list.append(metrics[1])
				f1_list.append(metrics[2])
				roc_auc_score_list.append(roc_auc_score(test,pred))

				tn, fp, fn, tp = confusion_matrix(test, pred).ravel()
				specificity_list.append(tn / (tn+fp))


		acc_avg = np.average(acc_list)
		acc_std = np.std(acc_list)

		prec_avg = np.average(prec_list)
		prec_std = np.std(prec_list)

		recall_avg = np.average(recall_list)
		recall_std = np.std(recall_list)

		f1_avg = np.average(f1_list)
		f1_std = np.std(f1_list)

		roc_auc_avg = np.average(roc_auc_score_list)
		roc_auc_std = np.std(roc_auc_score_list)

		specificity_avg = np.average(specificity_list)
		specificity_std = np.std(specificity_list)

		
		print(datatype)
		print("Accuracy: ", acc_avg, " +/- ", acc_std)
		print("Precision: ", prec_avg, " +/- ", prec_std)
		print("Recall: ", recall_avg, " +/- ", recall_std)
		print("F1: ", f1_avg, " +/- ", f1_std)
		print("AUC_ROC: ",roc_auc_avg," +/- ",roc_auc_std)   
		print("Specificity: ",specificity_avg," +/- ",specificity_std)



def LogReg_Baseline():

    """
    Implements a Logistic Regression baseline model for clinical event classification 
    using structured and optionally unstructured text data.

    Returns:
    None
    """

    # Configuration
	use_text = "" # Change to "_text" if using text features
	mode = "load" # Change to "save" to preprocess & save data
	task = "Dependence" # Choose from either "Dependence" or "Prescription" tasks

	batch_size = 10
	number_of_notes_per_patient = 40 
	number_of_note_splices = 10
	sequence_length = 200

	save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/saved_models/Final_Models" 
 
	# Data Loading
	training_params = {"batch_size": batch_size, "shuffle": True, "drop_last": False}
	test_params = {"batch_size": batch_size, "shuffle": False, "drop_last": False}

	training_set = MyDataSet("train",number_of_notes_per_patient,number_of_note_splices,task,sequence_length)
	training_generator = DataLoader(training_set, **training_params)

	test_set = MyDataSet("test",number_of_notes_per_patient,number_of_note_splices,task,sequence_length)
	test_generator = DataLoader(test_set, **test_params)

	# Extract Features and Labels
	X_train_structured, X_train_static, X_train_text, y_train = training_set.X_structured, training_set.X_static, training_set.X_text, training_set.y
	X_test_structured, X_test_static, X_test_text, y_test = test_set.X_structured, test_set.X_static, test_set.X_text, test_set.y


	if mode == "save":

		training_data = []
		training_labels = []
		for i in tqdm(range(len(X_train_static))): 
			
			current_instance = []

			# Encoding static features
			static_displacement = 100000
			current_static = X_train_static[i]
			current_static = [int(static_displacement + ((i+1) * a)) for i,a in enumerate(current_static) if a != 0]
			current_instance += current_static

			# Encoding structured features
			structured_displacement = 200000
			current_structured = X_train_structured[i]
			current_structured = [structured_displacement + a for a in current_structured if a != 0]
			current_instance += current_structured 

			# Encoding text features
			if use_text == "_text":

				text_displacement = 300000
				current_text = X_train_text[i]
				current_text = [item for sublist in current_text for item in sublist]
				current_text = [text_displacement + a for a in current_text if a != 0]
				current_instance += current_text


			current_instance = [str(a) for a in current_instance]
			training_data.append(" ".join(current_instance))
			training_labels.append(y_train[i]) 


		test_data = []
		test_labels = []
		for i in tqdm(range(len(X_test_static))):		
			current_instance = []

			static_displacement = 100000
			current_static = X_test_static[i]
			current_static = [int(static_displacement + ((i+1) * a)) for i,a in enumerate(current_static) if a != 0]
			current_instance += current_static


			structured_displacement = 200000
			current_structured = X_test_structured[i]
			current_structured = [structured_displacement + a for a in current_structured if a != 0]
			current_instance += current_structured


			if use_text == "_text":
				text_displacement = 300000
				current_text = X_test_text[i]
				current_text = [item for sublist in current_text for item in sublist]
				current_text = [text_displacement + a for a in current_text if a != 0]
				current_instance += current_text 


			current_instance = [str(a) for a in current_instance]
			test_data.append(" ".join(current_instance))
			test_labels.append(y_test[i]) 

		with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Data_Exploration/" + task + use_text,"wb") as f:
			pickle.dump((training_data,training_labels,test_data,test_labels),f)

	elif mode == "load":
		with open("/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Data_Exploration/" + task + use_text,"rb") as f:
			(training_data,training_labels,test_data,test_labels) = pickle.load(f)

	# Convert data to bag-of-words features
	cv = CountVectorizer(ngram_range=(1,3),max_features=20000)  
	training_data = cv.fit_transform(training_data)
	test_data = cv.transform(test_data)

	# Train Logistic Regression model
	clf = LogisticRegression()
	clf.fit(training_data[:,:],training_labels[:])


	y_pred = clf.predict(test_data)
	y_pred_proba = [a[1] for a in clf.predict_proba(test_data)]

	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	specificity = tn / (tn+fp)


	print("ACCURACY: ", accuracy_score(y_pred,y_test))
	print("PRECISION, RECALL, F1 SCORE: ",precision_recall_fscore_support(y_test,y_pred,average="weighted")) 
	print("SPECIFICITY",specificity) 
	print("AUC_ROC: ",roc_auc_score(y_test,y_pred_proba)) 




def brier_score():

    """
    Computes and plots the calibration curve for the predicted probabilities of two tasks 
    ('Prescription' and 'Dependence') using isotonic regression.

    Returns:
        None (Saves the calibration plot to disk).
    """

	save_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Plots/"

	seed = 8

	fig, ax = plt.subplots()


	for task in ["Prescription","Dependence"]:

		if task == "Prescription":
			test_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/Predictions/" + task + "_proba_" + str(seed)

		elif task == "Dependence":
			test_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/Predictions/" + task + "_proba_" + str(seed)

		val_path = "/home/kashyap/Pain_Project/Structured_Event_Classifier/Data/Prescription_plus_Dependence_Combined_Data/Predictions/" + task + "_proba_final_val_" + str(seed)

		# Load validation and test predictions
		with open(val_path,"rb") as f:
			(val_pred,val_label) = pickle.load(f) 

		with open(test_path,"rb") as f:
			(te_pred,te_label) = pickle.load(f) 

		print("BRIER SCORE", task, ":", brier_score_loss(te_label,te_pred[:,1])) 

		# Apply Isotonic Regression for probability calibration
		val_pred = np.array(val_pred)[:,1]
		isotonic = IsotonicRegression()
		isotonic.fit(val_pred, val_label)
		isotonic_probs = isotonic.predict(te_pred[:,1])

		# Compute calibration curve
		dl_y, dl_x = calibration_curve(te_label, isotonic_probs)

		# Plot calibration curve for the task
		plt.plot(dl_x,dl_y, marker='o', linewidth=1, label=task)

	# Plot reference line (perfect calibration)
	line = mlines.Line2D([0, 1], [0, 1], color='black')

	transform = ax.transAxes
	line.set_transform(transform)
	ax.add_line(line)
	fig.suptitle('Calibration Plot for the Deep Learning Models across both Tasks')
	ax.set_xlabel('Predicted probability')
	ax.set_ylabel('True probability in each bin')
	plt.legend()
	plt.savefig(save_path + "Calibration_Plot_final.png") 





if __name__ == "__main__":
	
	start = datetime.now()

	train()
	get_evaluation_metrics()      
	LogReg_Baseline()
	brier_score()  

	end = datetime.now()

	print("Total Run Time: ", end-start)

