from data_provider import alexhu_datasets

#provider the name of dataset (just  omit this step)

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width,image_channel, is_training=True):
    '''Given a dataset name and returns a Dataset.
    Args:
        dataset_name: String, the name of the dataset.
        train_data_paths: List, [train_data_path1, train_data_path2...]
        valid_data_paths: List, [val_data_path1, val_data_path2...]
        batch_size: Int
        img_width: Int
        img_channnel:Int
        is_training: Bool
    Returns:
        if is_training:
            Two dataset instances for both training and evaluation.
        else:    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
            One dataset instance for evaluation.
    Raises:
        ValueError: If `dataset_name` is unknown.
    '''

    train_data_list = train_data_paths
    valid_data_list = valid_data_paths

    if dataset_name :
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name+'test iterator'}
        test_input_handle = alexhu_datasets.InputHandle(test_input_param)
        test_input_handle.begin()
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name+' train iterator'}
            train_input_handle = alexhu_datasets.InputHandle(train_input_param)
            train_input_handle.begin()
            return train_input_handle, test_input_handle
        else:
            return test_input_handle
