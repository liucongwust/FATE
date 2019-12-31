## Homo Neural Network Configuration Usage Guide.

This section introduces the dsl and conf for usage of federated NCF([Neural Collaborative Filtering](http://dx.doi.org/10.1145/3038912.3052569) which 
was published in WWW'17).


We have provided upload config for you can upload example data conveniently.

#### Upload Data
citeulike_a data set 
1. Guest Party Data: upload_data_guest.json 
2. Host Party Data: upload_data_host.json

#### Training Task.
dsl: test_hetero_ncf_train.dsl
runtime_config : test_hetero_ncf.json
  
#### Training and Evaluation Task.
dsl: test_hetero_ncf_train_then_predict.dsl
runtime_config : test_hetero_ncf_train_then_predict.json
   
Users can use following commands to running the task.
 ```bash
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}
```   

Moreover, after successfully running the training task, you can use it to predict too.
