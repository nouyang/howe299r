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


#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h>
#endif

#define PIN 6
//Adafruit_BMP280 bme; // I2C
//Adafruit_BMP280 bme(BMP_CS); // hardware SPI
Adafruit_BMP280 bme(BMP_CS, BMP_MOSI, BMP_MISO,  BMP_SCK);
Adafruit_BMP280 bme2(BMP_CS2, BMP_MOSI, BMP_MISO,  BMP_SCK);
Adafruit_BMP280 bme3(BMP_CS3, BMP_MOSI, BMP_MISO,  BMP_SCK);

unsigned long scalenorm = 0;
unsigned long scaleunit = 0;
int prevnumpix = 0;
// Parameter 1 = number of pixels in strip
// Parameter 2 = Arduino pin number (most are valid)
// Parameter 3 = pixel type flags, add together as needed:
//   NEO_KHZ800  800 KHz bitstream (most NeoPixel products w/WS2812 LEDs)
//   NEO_KHZ400  400 KHz (classic 'v1' (not v2) FLORA pixels, WS2811 drivers)
//   NEO_GRB     Pixels are wired for GRB bitstream (most NeoPixel products)
//   NEO_RGB     Pixels are wired for RGB bitstream (v1 FLORA pixels, not v2)
//   NEO_RGBW    Pixels are wired for RGBW bitstream (NeoPixel RGBW products)
Adafruit_NeoPixel strip = Adafruit_NeoPixel(60, PIN, NEO_GRB + NEO_KHZ800);

unsigned long avgbme;
unsigned long avgbme2;

void setup() {
    Serial.begin(115200);
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'   
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
    
    scalenorm = bme.readPressure();
  
    scaleunit = 0.001 * scalenorm; // we expect a light pressure to be this much

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
    Serial.print(avgbme/10);
    Serial.print(" ");
    Serial.println(avgbme2/10);
    /*Serial.print(bme.readPressure());*/


    unsigned long centered = (avgbme/10)-scalenorm;
    unsigned long numpixlong = map(centered,0,scaleunit*150, 0, 10); // maximum 10 pixels light up

    numpixlong = constrain(numpixlong, 0, 10);
    int numpix = (int) numpixlong;
    //Serial.print("numpix: ");
    //Serial.println(numpix);
    if (prevnumpix >= numpix){
          for(uint16_t i=numpix; i<= prevnumpix; i++) {
              strip.setPixelColor(i, strip.Color(0, 0, 0, 255)); //White
          }
    }
    else{
      for(uint16_t i=0; i< numpix; i++) {
          strip.setPixelColor(i, strip.Color(0, 100, 10)); //R, G, B
      }}
  strip.show();
  prevnumpix = numpix;
  delay(DELAY);
}

// Fill the dots one after the other with a color
void colorWipe(uint32_t c, uint8_t wait) {
  for(uint16_t i=0; i<strip.numPixels(); i++) {
    strip.setPixelColor(i, c);
    strip.show();
    delay(wait);
  }
}

