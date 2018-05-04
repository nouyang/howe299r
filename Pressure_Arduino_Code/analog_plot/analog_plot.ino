/***************************************************************************
@ nouyang
04 May 2018

Arduino side of arduino-python real-time plotting
See analog-plot.py for matching python code
Prints to serial port, time (milliseconds) and pressure in Pa (from BMP280 via SPI)


BMP280: Uses Adafruit BMP280 library
Plotting: Uses code from electronut.in
***************************************************************************/
/***************************************************************************
  This is a library for the BMP280 humidity, temperature & pressure sensor

  Written by Limor Fried & Kevin Townsend for Adafruit Industries.  
  BSD license, all text above must be included in any redistribution
 ***************************************************************************/

#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <Time.h>

#define BMP_SCK 13
#define BMP_MISO 12
#define BMP_MOSI 11 
#define BMP_CS 10
#define BMP_CS2 9
#define BMP_CS3 8

#define DELAY 50 
#define SMALLDELAY 5 

//Adafruit_BMP280 bme; // I2C
//Adafruit_BMP280 bme(BMP_CS); // hardware SPI
Adafruit_BMP280 bme(BMP_CS, BMP_MOSI, BMP_MISO,  BMP_SCK);
Adafruit_BMP280 bme2(BMP_CS2, BMP_MOSI, BMP_MISO,  BMP_SCK);
Adafruit_BMP280 bme3(BMP_CS3, BMP_MOSI, BMP_MISO,  BMP_SCK);

unsigned long avgbme;
unsigned long avgbme2;

void setup() {
    Serial.begin(115200);
  /*Serial.println(F("BMP280 test"));*/
  /*if (!bme3.begin()) {  */
    /*Serial.println("Could not find a valid BMP280 #3 (pin 8) sensor, check wiring!");*/
    /*while (1);*/
  /*}*/
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
    /*Serial.print("B1 Adafruit Breakout Pressure = ");*/
    avgbme = 0;
    avgbme2 = 0;
    for (int i=0; i<10; i++){ 
        avgbme += bme.readPressure();
        avgbme2 += bme2.readPressure();
        delay(SMALLDELAY);
        
    }
    /*Serial.print(bme.readPressure());*/
    Serial.print(avgbme/10);
    Serial.print(" ");
    Serial.println(avgbme2/10);
    /*Serial.println(bme2.readPressure());*/
    /*Serial.print(" ");*/
    /*Serial.println(" Pa");*/


    /*[>Serial.print("B2 Generic Board Pressure = ");<]*/
    /*Serial.print(bme2.readPressure());*/
    /*[>Serial.println(" Pa");<]*/
    /*delay(DELAY);*/

    /*[>Serial.print("B3 Generic Board Pressure = ");<]*/
    /*Serial.print(bme3.readPressure());*/
    /*[>Serial.println(" Pa");<]*/


    /*delay(DELAY);*/
}
