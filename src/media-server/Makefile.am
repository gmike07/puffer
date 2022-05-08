AM_CPPFLAGS = $(CXX17_FLAGS) $(SSL_CFLAGS) $(POSTGRES_CFLAGS) \
	-I$(srcdir)/../util -I$(srcdir)/../net -I$(srcdir)/../notifier \
	-I$(srcdir)/../monitoring -I$(srcdir)/../abr \
	-isystem$(srcdir)/../../third_party/json.upstream/single_include/nlohmann \
	-isystem$(srcdir)/../../third_party/libtorch/include \
	-isystem$(srcdir)/../../third_party/libtorch/include/torch/csrc/api/include \
    -I/usr/include/python3.7 \
	-I/home/ubuntu/.local/lib/python3.7/site-packages/pybind11/include
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(EXTRA_CXXFLAGS)


noinst_LIBRARIES = libnethelper.a

# libnet_a_LDFLAGS = -L../../third_party/libtorch/lib \
# 	'-Wl,-rpath,$$ORIGIN/../../third_party/libtorch/lib'

# libnet_a_LDADD = $(POSTGRES_LIBS) $(SSL_LIBS) $(CRYPTO_LIBS) $(YAML_LIBS) -lstdc++fs \
#                     -ltorch -lcaffe2 -lc10 -lmkldnn -lcurl -lcurlpp -lpthread
 

libnethelper_a_SOURCES = ws_client.hh ws_client.cc channel.hh channel.cc \
	client_message.hh client_message.cc server_message.hh server_message.cc \
	../notifier/inotify.hh ../notifier/inotify.cc \
	../abr/abr_algo.hh ../abr/abr_algo.cc \
	../abr/linear_bba.hh ../abr/linear_bba.cc \
	../abr/mpc.hh ../abr/mpc.cc \
	../abr/mpc_search.hh ../abr/mpc_search.cc \
	../abr/pensieve.hh ../abr/pensieve.cc \
	../abr/puffer.hh ../abr/puffer.cc \
	../abr/puffer_raw.hh ../abr/puffer_raw.cc \
	../abr/puffer_ttp.cc ../abr/puffer_ttp.hh \
	../abr/puffer_reinforce.cc ../abr/puffer_reinforce.hh \
	../abr/reinforce_policy.cc ../abr/reinforce_policy.hh \
	../abr/puffer_exp3.cc ../abr/puffer_exp3.hh 
	../../third_party/json.upstream/single_include/nlohmann/json.hpp


bin_PROGRAMS = run_servers maintenance_server ws_media_server


ws_media_server_SOURCES = ws_media_server.cc
ws_media_server_LDFLAGS = -L../../third_party/libtorch/lib \
	'-Wl,-rpath,$$ORIGIN/../../third_party/libtorch/lib'
ws_media_server_LDADD = ../util/libutil.a ../net/libnet.a ./libnethelper.a ../util/libutil.a \
	$(POSTGRES_LIBS) $(SSL_LIBS) $(CRYPTO_LIBS) $(YAML_LIBS) -lstdc++fs \
	-ltorch -lcaffe2 -lc10 -lmkldnn -lcurl -lcurlpp -lpthread 

run_servers_LDFLAGS = -L../../third_party/libtorch/lib \
	'-Wl,-rpath,$$ORIGIN/../../third_party/libtorch/lib'
run_servers_SOURCES = run_servers.cc
	../monitoring/influxdb_client.hh ../monitoring/influxdb_client.cc
run_servers_LDADD = ../util/libutil.a ../net/libnet.a ./libnethelper.a ../util/libutil.a \
	$(POSTGRES_LIBS) $(SSL_LIBS) $(CRYPTO_LIBS) $(YAML_LIBS) -lstdc++fs \
	-ltorch -lcaffe2 -lc10 -lmkldnn -lcurl -lcurlpp -lpthread 

maintenance_server_LDFLAGS = -L../../third_party/libtorch/lib \
	'-Wl,-rpath,$$ORIGIN/../../third_party/libtorch/lib'
maintenance_server_SOURCES = maintenance_server.cc \
	server_message.hh server_message.cc
maintenance_server_LDADD = ../util/libutil.a ../net/libnet.a ./libnethelper.a ../util/libutil.a \
	$(SSL_LIBS) $(CRYPTO_LIBS) $(YAML_LIBS) -lstdc++fs \
	-ltorch -lcaffe2 -lc10 -lmkldnn -lcurl -lcurlpp -lpthread 


# train_nn_LDFLAGS = -L../../third_party/libtorch/lib \
# 	'-Wl,-rpath,$$ORIGIN/../../third_party/libtorch/lib' -O3
# train_nn_SOURCES = train_nn_try.hh train_nn_try.cc tqdm.hpp
# train_nn_LDADD = ../util/libutil.a  $(YAML_LIBS) -lstdc++fs \
# 	-ltorch -lcaffe2 -lc10 -lmkldnn -lcurl -lcurlpp -lpthread -lboost_program_options -lboost_system -lboost_filesystem
