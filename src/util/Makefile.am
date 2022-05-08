AM_CPPFLAGS = -I$(srcdir)/../net $(CXX17_FLAGS) \
	-isystem$(srcdir)/../../third_party/json.upstream/single_include/nlohmann \
	-isystem$(srcdir)/../../third_party/libtorch/include \
	-isystem$(srcdir)/../../third_party/libtorch/include/torch/csrc/api/include \
	-I$(srcdir)/../abr -I$(srcdir)/../notifier -I$(srcdir)/../media-server

AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(EXTRA_CXXFLAGS)

noinst_LIBRARIES = libutil.a

libutil_a_SOURCES = \
	child_process.hh child_process.cc \
	exception.hh \
	file_descriptor.hh file_descriptor.cc \
	path.hh path.cc \
	pipe.hh pipe.cc \
	poller.hh poller.cc \
	signalfd.hh signalfd.cc \
	strict_conversions.hh strict_conversions.cc \
	system_runner.hh system_runner.cc \
	temp_dir.hh temp_dir.cc \
	temp_file.hh temp_file.cc \
	timeit.hh timeit.cc \
	timestamp.hh timestamp.cc \
	timerfd.hh timerfd.cc \
	tokenize.hh tokenize.cc \
	formatter.hh formatter.cc \
	util.hh util.cc \
	filesystem.hh \
	chunk.hh \
	mmap.hh mmap.cc \
	y4m.hh y4m.cc \
	ipc_socket.hh ipc_socket.cc \
	pid.hh pid.cc \
	media_formats.hh media_formats.cc \
	yaml.hh yaml.cc \
	reinforce.hh reinforce.cc \
	sender.hh sender.cc \
	exp3.cc ex3.hh \
	context.cc context.hh \
	exp3_kmeans.cc exp3_kmeans.hh

libutil_a_LDFLAGS = -L../../third_party/libtorch/lib \
	'-Wl,-rpath,$$ORIGIN/../../third_party/libtorch/lib' \
	../../third_party/libtorch/lib/libtorch.so \
	../../third_party/libtorch/lib/libcaffe2.so \
	../../third_party/libtorch/lib/libc10.so  -g

