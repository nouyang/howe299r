/***************************************************************************
  This is a library for the BMP280 humidity, temperature & pressure sensor

  Designed specifically to work with the Adafruit BMEP280 Breakout 
  ----> http://www.adafruit.com/products/2651

  These sensors use I2C or SPI to communicate, 2 or 4 pins are required 
  to interface.

  Adafruit invests time and resources providing this open source code,
  please support Adafruit andopen-source hardware by purchasing products
  from Adafruit!

  Written by Limor Fried & Kevin Townsend for Adafruit Industries.  
  BSD license, all text above must be included in any redistribution
 ***************************************************************************/

#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>

#define BMP_SCK 13
#define BMP_MISO 12
#define BMP_MOSI 11 
#define BMP_CS 10
#define BMP_CS2 9
#define BMP_CS3 8

#define DELAY 50 

//Adafruit_BMP280 bme; // I2C
//Adafruit_BMP280 bme(BMP_CS); // hardware SPI
Adafruit_BMP280 bme(BMP_CS, BMP_MOSI, BMP_MISO,  BMP_SCK);
Adafruit_BMP280 bme2(BMP_CS2, BMP_MOSI, BMP_MISO,  BMP_SCK);
Adafruit_BMP280 bme3(BMP_CS3, BMP_MOSI, BMP_MISO,  BMP_SCK);

void setup() {
  Serial.begin(115200);
  Serial.println(F("BMP280 test"));
  if (!bme3.begin()) {  
    Serial.println("Could not find a valid BMP280 #3 (pin 8) sensor, check wiring!");
    while (1);
  }
  if (!bme2.begin()) {  
    Serial.println("Could not find a valid BMP280 #2 (pin 9) sensor, check wiring!");
    while (1);
  }
  if (!bme.begin()) {  
    Serial.println("Could not find a valid BMP280 sensor (pin 10), check wiring!");
    while (1);
  }
}
  
void loop() {
//    Serial.print("Temperature = ");
//    Serial.print(bme.readTemperature());
//    Serial.println(" *C");
    
    /*Serial.print("B1 Adafruit Breakout Pressure = ");*/
    Serial.print(bme.readPressure());
    /*Serial.println(" Pa");*/
    delay(DELAY);

    /*Serial.print("B2 Generic Board Pressure = ");*/
    Serial.print(bme2.readPressure());
    /*Serial.println(" Pa");*/
    delay(DELAY);

    /*Serial.print("B3 Generic Board Pressure = ");*/
    Serial.print(bme3.readPressure());
    /*Serial.println(" Pa");*/


    delay(DELAY);
//
//    Serial.print("Approx altitude = ");
//    Serial.print(bme.readAltitude(1013.25)); // this should be adjusted to your local forcase
//    Serial.println(" m");
//    
}
