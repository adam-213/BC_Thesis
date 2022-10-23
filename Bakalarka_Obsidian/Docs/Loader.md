Documentation for class Dataset
===

## PyTorch
Since I'm going to be using pyTorch for the neural network, it is mandatory to use their Dataloader class. 

### Dataloader basics
Data loader class makes it possible to not have all of the data in memory at the same time. To do this We only need to define 3 methods - 
1) Init 
2) Len
3) getItem - in this method we can load data from the hard drive and do some standardization operations on them.
	Then return them to be passed to the neural networks or other pyTorch components.

Interesting observation instead of defining next for the iterator, pytorch asks for len of the dataset and getItem(index) to be defined


### Dataset class
In addition to the basic methods, I'll be defining a few more to make it easier to read and understand the loading code, since we are dealing with 6 files per scan.
There is a very real possibility that I'll be combining scans from multiple datasets at random to make the #NN better at generalizing. 

- \_\_init\_\_ 
	Initialize self.scans for storing all the scan paths.
	I decided to mix all datasets together, with the hopeful goal of making the #NN better at generalizing. However I did it in a quick and dirty way so its would take some rewriting before it would be able to work with just a single dataset.
	However a single dataset does provide only a single run of the generator, which should not be sufficient enough to train effective #NN 

- load_scan_paths
	Firstly we find the \$Bakalarka\$ folder, which is the root folder, from there we concatenate path to the folder containing the datasets.
	Then we iterate over the folder, so we get a list of datasets, these subfolders contain scene_paths file.
	This file conveniently contains all the scan paths from that subfolder.
	So we read this file and then combine it with our existing absolute path. To get absolute path to each scan. However since each scan "contains" multiple files We leave the path vague:
	*"Bakalarka/DataSets/EXR/DS_03_01/captures/scan_127"* 
	this is to facilitate easily adding extensions based on the actual file being loaded

- paths
	as mentioned above we are using 6 files for each scan. from the general path to each scan we need to make it specific for each file
	since the filenames are "standardized" we just concatenate the general path to the specific file suffix.

- scan_load 
	calls paths to establish the file/s that its going to be loading and then 1by1 calls the functions to load that specific file type
	returns a tuple of these loaded files

- transforms - this might not be necessary because if I remember correctly this is applied in the actual training procedure 
	could be applied here and then the training procedure would be cleaner
	#todo

- getitem 
	one of the mandatory functions for the dataloader pyTorch class.
	I wanted to do a selection random, but it is not neccesary afaik, because the training procedures, do this automatically.
	read a path from the pile of scan path on an requested index. load it (this prevents the need to load gigabytes of data into RAM, can be loaded from disk - ssds are more than fast enough to read the 20 or megabytes per scan quite quickly).
	
	Quickly in this context means about 300ms which is a lot more than i was expecting. And will affect the performance of the training portion of the #NN question is how much.
		Pytorch apparently provides a prefetch something, will have to investigate that.
		Pytorch also provides torch.nn.botleneck to profile changes to loading speeds and bottlencks 
		#todo 

	also apply transfroms method  if needed #todo 

- len 
	We need to know the size of the dataset, for purposes of counting epochs and other pyTorch related things. (IDK them yet #todo )

----
### file specific loading functions 

Right now these don't do anything special, basic read of the filetype and return to the scan_load method of the dataset class 
#todo if something changes