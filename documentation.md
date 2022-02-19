# Prediction of Floor plan  
The pipeline is based on `3D Semantic Segmentation` of point clouds. The following steps occur in the pipeline:  
+ **Loading pre-trained model for semantic segmentation**: Firstly, we need to prepare our sub-pipeline i.e. 3D semantic segmentation. The main library that serves this purpose (3D object detection and 3D semantic segmentation is `Open3d ML`. Currently, only two models are available for these tasks (KPConv and RandLANet). For our suitable needs, the `RandLANet` model pre-trained on the `S3DIS` dataset is used. This is because it has the highest `imOu` score for semantic segmentation on indoor point clouds. To prepare the pipeline, we load the config file `randlanet_s3dis.yml` that contains config info about loading/ training/ testing RandLANet on the s3dis dataset if needed. After this, we check if there is any existing `checkpoint` file, if no checkpoint is found the code downloads the checkpoint by establishing an HTTP request. After these steps, a 3D semantic segmentation pipeline with appropriate configs and weights for the RandLANet model trained on the s3dis dataset is loaded.  

+ **Preparing point cloud data for semantic segmentation pipeline**: The next step is to prepare the data properly. If the tensors of the point cloud data to be fed into the pipeline aren't configured into the appropriate shape then we get an error. For the `randlanet_s3dis` model, the point cloud data should be in `dictionary` format. The point cloud dictionary for the model should have `3 attributes` i.e. `point`, `features`, and `label`. Point refers to the `x, y, and z` coordinates of point clouds in 3D space. Features refer to the `R, G, B` values of each point cloud. The label is the criteria that must be included for every purpose (infer, train, test). In the case of prediction on custom data, point clouds are given a label of `zero` initially.

+ **Initializing Open3d ML visualizer**: The open3d ml visualizer differs slightly from the native open3d visualizer. The pipeline invokes the ml-3d visualizer. To set labels of prediction (floors, ceilings, etc..) we also instantiate a `label lut` class. The label lut is used to add and set labels to a parent category. We manually specify s3dis_labels where `0 labels = Unlabeled` i.e. all point clouds predicted as 0 by the model were not identified by the model properly. These inferred point clouds can be fed into the Open3d ML visualizer class.  

+ **Running inference**: The point cloud dictionary is fed into the pre-trained model which runs inference on it. It classifies every point cloud into `14 labels`, if the model is not sure about certain point clouds then it classifies it into the `0/unlabeled` label. The `prediction` returned by the model contains two attributes: `label` and `confidence score`. Thus, every point cloud dictionary is given a new attribute `prediction` that contains the label to which the point cloud belongs to.  

+ **Fitting a 2D plane**: Next, we disintegrate and extract the required labels from the predicted labeled point clouds. For example, to extract a floor plan, we are interested in `floor, walls, windows, columns and doors`. The points and features of these labels are stored in separate arrays respectively. To apply open3d functionalities, all these points and features parameters are compiled to convert them into open3d `pcd` class format. Finally, we fit a plane into `floor pcds` using `RANSAC` estimation. This returns a plane in format `ax+by+cz+d=0` which is later utilized for perspective projection purposes.  

+ **Project 3D point clouds into 2D horizontal plane and extracting floor plan**: Since we are dealing with upright models, we can project the 3D pcds into the floor plane by simply scattering the points in a 2D plane by using their `x` and `y` coordinates only using `matplotlib`. This current method will not give a consistent floor plan if the model is inclined (floor is not parallel to xy-plane). The final result is then saved as a `png` file into the desired output directory.  

+ **Visualizations**: The pipeline provides the option of turning visualization parameters `on` and `off` by default these options are set to `False`. We can visualize the semantic segmentation results on point clouds from the open3d ml sub pipeline from the open3d ml visualizer. We can also visualize the results after fitting the plane and extracting the necessary labels from pcds from native open3d pcds.  

## **Note:** 
+ The `PLY` file to be fed into the pipeline must contain `vertex` properties of `x, y, z, Red, Green and Blue` and `edge` properties are discarded by pipeline. 
+ It is to be noted that `pcd` contains points in NumPy array float64 data format and colors normalized in the range [0,1] in uint8 data type.
+ For matplotlib, the `TkAgg` engine is used instead of `qt` because qt has some version compatibility issues that conflict with the pipeline. Install the required `TkAgg` engine from the terminal to avoid any errors.  

## Usage:
### Planar Cloud Extraction
Use this code to store the planar/non-planar components of input point cloud. The code is available in `script/segmentation.py`
#### Mandatory Arguments  
+ `--data_path` : The path to input point cloud
+ `--output_dir` : The directory to store point clouds of floor, ceiling, walls and others in separate PLY files.  
#### Optional Arguments 
+ `--task` : Should be kept `extraction` (default) for this task
+ `--vis_prediction`: Option to visualize semantic segmentation result. 
+ `--vis_open3d`: Option to visualize floor plan in 3D from open3d.
#### Example
```
python segmentation.py --data_path data/conferenceRoom_1.ply --output_dir output/ --task floor_plan
```
### Floor Plan Extraction
The code is available in `script/segmentation.py`
#### Mandatory Arguments  
+ `--data_path` : The path to input point cloud
+ `--output_dir` : The directory to store floor plan in png format.  
+ `--task` : Should be kept `floor_plan` for this task
#### Optional Arguments 
+ `--vis_prediction`: Option to visualize semantic segmentation result. 
+ `--vis_open3d`: Option to visualize floor plan in 3D from open3d. 
#### Example
```
python segmentation.py --data_path data/conferenceRoom_1.ply --output_dir output/ --task floor_plan
```
### Planar Mesh Reconstruction
Use the individual point clouds (.ply) of floor, ceiling, walls and others extracted using `segmentation.py` as described above to generate a decimated mesh.
#### Usage
```
usage: reconstruct.py [-h] [-iw WALLS_FILE] [-if FLOOR_FILE] [-ic CEILING_FILE] [-io OTHERS_FILE] [-o OUTPUT] [-m MESHING_ALGORITHM] [--no_projection] [--no_segmentation]

optional arguments:
  -h, --help            show this help message and exit
  -iw WALLS_FILE, --walls_file WALLS_FILE
                        Input point cloud of just walls
  -if FLOOR_FILE, --floor_file FLOOR_FILE
                        Input point cloud of floor
  -ic CEILING_FILE, --ceiling_file CEILING_FILE
                        Input point cloud of ceiling
  -io OTHERS_FILE, --others_file OTHERS_FILE
                        Input point cloud of other points
  -o OUTPUT, --output OUTPUT
                        Output mesh file path
  -m MESHING_ALGORITHM, --meshing_algorithm MESHING_ALGORITHM
                        poisson(default), ball_pivot, alpha_shapes
  --no_projection       Do not project all points to plane
  --no_segmentation     Do not process walls individually
```
#### Example
```
python reconstruct.py -iw output/walls.ply -if data/floor.ply -ic data/ceiling.ply -io data/others.ply -o data/out.ply --no_segmentation
```
