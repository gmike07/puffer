cd ..
cd mahimahi
sudo apt-get install libprotobuf-dev protobuf-compiler -y
sudo apt install apache2-dev -y
sudo apt install xcb -y
sudo apt-get install libx11-dev -y
sudo apt-get install libxcb-xrm-dev -y 
sudo apt-get install libxcb-present-dev -y
sudo apt-get install -y libsdl-pango-dev -y 
sudo apt install git-buildpackage -y
sudo ./autogen.sh
sudo ./configure
cd src
sudo make
cd frontend
for file in *; do
	if [[ -x "$file" ]]
	then
            sudo chown root:root $file
	    sudo chmod 4755 $file
	fi
done
cd ..
cd ..
cd ..
