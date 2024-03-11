import gurobi_logtools as glt

summary = glt.parse("./log_file_cg.log").summary()
summary()

Log2Console =0