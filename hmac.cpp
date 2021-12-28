#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <string>
using std::string;

#include "cryptlib.h"
using CryptoPP::Exception;

#include "hmac.h"
using CryptoPP::HMAC;

#include "sha.h"
using CryptoPP::SHA256;

#include "base64.h"
using CryptoPP::Base64Encoder;

#include "filters.h"
using CryptoPP::StringSink;
using CryptoPP::StringSource;
using CryptoPP::HashFilter;

string sign(string key, string plain)
{
	string mac, encoded;
	try
	{
		HMAC< SHA256 > hmac((byte*)key.c_str(), key.length());		

		StringSource(plain, true, 
			new HashFilter(hmac,
				new StringSink(mac)
			) // HashFilter      
		); // StringSource
	}
	catch(const CryptoPP::Exception& e)
	{
		cerr << e.what() << endl;
	}

	encoded.clear();
	StringSource(mac, true,
		new Base64Encoder(
			new StringSink(encoded)
		) // Base64Encoder
	); // StringSource

	return encoded;
}