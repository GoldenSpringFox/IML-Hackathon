# IML Hackathon 2026

Bike Dataset: https://drive.google.com/drive/folders/1Y6tNS9r65jRkl5eQXgA5KSWWGI1rRK6j?usp=drive_link

## Opening Notes
We're CTO of a startup aiming to develop one of these:
1. The Chain Rule: a bike share company serving cities world wide
2. PixelPerfect: most robust image object detection models

Problems:
- both have real world data: noise, missing data, weird problems. 
- lets us be creative: these are open-ended challenges. we can't "get the right answer"
- do our best, use whatever to solve the problem. can get more data online or anything else we need

Format:
- full details in moodle
- submit until 11am friday
- feel free to ask lecturers and TAs physically
- hackathon forums will be occasionally answered
- notice moodle announcement
- task difficulty will be taken into consideration

Time Management:
- select project and decide on start and end time
- block ~5 hours to prepare submissions
- start immediately with a working baseline: easiest model we can think of, evaluate using official scripts, and submit it!
- after that iterate: clean data, explore fancier models, tune hyperparameters
- stages: preprocessing -> data exploration -> baseline -> model
- don't overdo it! we don't have to use all the time.

*all hours are collective human hours, so 5 ~ 1.5

Tips:
- work iteratively
- decide on a data release schedule
- use github
- use command line arguments (docopt or argparse)
- read instructions carefully from top to bottom at the start
- learn on the way: read, explore.. but don't copy other people's code. also don't copy weights from others
- don't get blocked! just make reasonable assumptions and move on
- don't give up, ML in practice is mostly failing alot
- measure each model's performance to previous models. think of metrics to do so.

Interview:
- no need to make presentation or memorize everything we've done
- just know why we made our desicions and talk about our solution

### Task 1: Bikes
- cities have different demands
- we want a model that generalizes all possible cities and predicts bike demand
- the model should be good enough to handle a new city given very few samples from that city, and a city with no data at all
- unusual regression model: more than one underlying distribution, interesting how to split the data

### Task 2: Image Classification
- 20,000 images
- the challenge is making the model robust, so handle weird backgrounds / rotations / filters etc..
