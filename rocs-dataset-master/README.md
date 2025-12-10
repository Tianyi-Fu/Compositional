This contains all generated property data and conceptual knowledge generated from the property data for RoCS project. Please refer to our paper "RoCS - Robot-centric Object Knowledge Acquisition Framework" for the detailed explaination of property definitions, acquisition methods and conceptual knowledge generation. 

## File Structure
There is one csv file per property. 
Each files contains 1100 data points, namely 10 repetitions for all object instance. 

Each file has the first column instance_name as "_object\_class\_#number\_#repetition_". The following columns contain the value of the particular property. 
If the values are multidimensional, each dimension will be shown in one comma seperated column. 

The directory "conceptual_knowledge" contains the conceptual knowledge about objects. The directory has two json files:
1) File "object_instance_knowledge.json" contains knowledge about 110 object instances belonging to 11 object classes.
2) File "object_concept_knowledge.json" contains knowledge about 11 object classes.

* This is an illustration of a conceptual knowledge about objects where symbols are generated using k-means clustering method. Please refer to our paper "RoCS - Robot-centric Object Knowledge Acquisition Framework" for a detailed discussion on the knowledge base generation.

## Object Classes
We consider 11 different object classes where each class consists of 10 unique object instances. That leads to a total number of 110 object instances.

The object classes are namely:
*  Ball
*  Book
*  Bowl
*  Cup
*  Metal Box
*  Paper Box
*  Plastic Box
*  Plate
*  Sponge
*  To Go Cup
*  Tray
