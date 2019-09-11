import tensorflow as tf

reader = tf.train.NewCheckpointReader('out/model.ckpt')
all_variables = reader.get_variable_to_shape_map()
all_variables_type = reader.get_variable_to_dtype_map()
print(all_variables)
print(all_variables_type)

for variable_name in all_variables:
    print(variable_name, 'shape is :', all_variables[variable_name])

print('Value for variable v1 is :', reader.get_tensor('v1'))
print('Value for variable v2 is :', reader.get_tensor('v2'))
