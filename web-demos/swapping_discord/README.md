# Using Midjourney and the Picsi.AI by InsightFaceSwap Bot to create a personalized portrait

We have named this highly realistic portrait creation tool as ``Picsi.AI``. You can use it for free, or head over to [Patreon](https://www.patreon.com/picsi) to subscribe and access more features and higher usage limits.

## Important Links

1. Discord bot invitation link: https://discord.com/api/oauth2/authorize?client_id=1090660574196674713&permissions=274877945856&scope=bot
2. Discord discussion server(to get help): https://discord.gg/Ym3X8U59ZN
3. Patreon subscription: https://www.patreon.com/picsi

## ChangeLog

**`2023-08-27`** 
1) **Enhanced GIF Quality**: Improved resolution, reduced noise, and enhanced sharpness for GIF outputs. This aims to provide a clearer and better visual experience.
2) **Increased GIF Size Limit**: The maximum allowable GIF file size has been increased from 7MB to 10MB, allowing for more detailed and creative GIFs.
3) **Extended Frame Limit for Pro Members**: Pro members can now utilize up to 75 frames for GIFs at a flat rate of 30 credits. This expands the possibilities for more complex and intricate GIFs.
4) **GIF Support for Basic Members**: Basic members now have access to GIF support, limited to 20 frames at a cost of 20 credits.
5) **URL Support for GIF**: Added the ability to directly work on GIFs using URL links, eliminating the need to download and re-upload GIF files. Provides an easier and faster way to create funny GIFs.
6) For examples in detail, please jump to https://www.patreon.com/posts/88351201.



**`2023-08-25`** 
 Time Travel Has Never Been So Easy! Introducing Oldifying Faces.
  1) Use a saved face and transfer it into your target image, then apply the oldifying effect. For instance:
     
       ``/swapid johndoe --oldify 300``
     
     This will take the saved face named johndoe, and then oldify it with an intensity of 300.
     
     Note that we can use ``-o`` as a shorthand for the ``--oldify`` argument.
  3) You can directly oldify a face in the attached picture without transfer it with one of your saved faces:
     
       ``/swapid _ --oldify 200``
     
  4) Use the --oldify option to set the transformation intensity, ranging from 1 to 1000. The default intensity is 300 if none is specified.

       ``/swapid _ --oldify``
     
  5) Special reminder: Due to the additional arguments parsing, please make sure that the input for idname does not contain any spaces. For example, ``/setid A,B`` is allowed, but ``/setid A, B`` is incorrect.
      <div align="left">
         <img src="https://github.com/nttstar/insightface-resources/blob/master/images/v0.3_image.jpg?raw=true" width="640"/>
      </div>

**`2023-08-02`** 
  We have deployed a new model and optimized three aspects:
  1) The new model performs better in handling skin shading under complex lighting conditions, reducing the likelihood of generating black or white erroneous pixels on the skin.
  2) We have optimized the handling of glasses in the Saved/Source photo. When the Source photo contains glasses, we will generate the image based on the version without glasses to avoid any ghosting effects caused by glasses in the resulting photo. For target images that originally have glasses or sunglasses, this process will not affect the final results.
  3) We have optimized the handling of bangs/fringe. When the source photo has thick bangs/fringe, we will try to minimize the impact on the generated result.
      <div align="left">
         <img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/picsi_20230802.jpg" width="640"/>
      </div>

**`2023-06-02`**
  1) The length limit for idname has been increased to 10, and the maximum number of idnames that can be saved has been increased to 20.
  2) Remove the 'greedy' prefer option, now '--nogreedy' and '--greedy' produce the same result.
  3) The feature of ID mixing has been added. You can use the symbol "+" to link multiple idnames (up to 3) to generate interesting results. For example, ``/setid father+mother`` might generate an image similar to their son, and ``/setid mother+father`` might generate a photo like their daughter (that is, the order of the "+" link will affect the result). You can also use ``/setid mother+mother+father`` to enhance the features of the mother ID. There's an example [here](https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/240_14.jpeg)

**`2023-05-17`**
  1) The maximum command usage per image is set to 2, meaning that even if there are 4 faces in a single image, it will only consume 2 commands.
  2) Now we use a queue in our backend. When there are too many users online, the requests will be queued and processed one by one which may slow to respond.
  3) The support for GIFs has been temporarily removed, in order to ensure fast response time.

**`2023-05-13`**
  Now we support the **greedy** mode as the default option, which can provide higher identity similarity. You can use the ``/setid --nogreedy``(put ``--nogreedy`` in the ``idname`` field) command to disable it (and use ``/setid --greedy`` to enable again). In addition, the ``/listid`` command can be used to view the current ID name and prefer options. For more information, please refer to the instruction of the ``/setid`` command on this page.

**`2023-05-08`**
  1) The maximum pixel output has now been changed to 2048, previously it was 1920.
  2) The number of command statistics have been changed from the number of images to the number of faces (i.e. if there are 2 faces in one image, it will consume 2 commands).

**`2023-04-27`**
  1) Now we support swapping on GIFs. The usage is the same as static images. A few extra key points: 1) Uploaded gifs cannot exceed 5MB in size; 2) Performing one gif face swap will consume 5 command opportunities (i.e. a maximum of 10 gifs can be operated per day); 3) Up to the first 15 frames can be operated; 4) Supports single-person swapping only in GIFs; 5) The frames may be dynamically resized to a lower resolution.
  2) Add FAQ.

**`2023-04-18`**
  Now we support Discord application commands(AKA. slash commands), please remember joining our [Discord group](https://discord.gg/Ym3X8U59ZN) to get notification.

## Disclaimer

By using this service, you acknowledge that you have read, understood, and agreed to the terms and conditions outlined in this disclaimer.

We would like to emphasize that our service is intended for research and legal AI creation purposes only. We do not condone or promote the use of our service for any illegal or unethical activities. We strictly prohibit the use of our service to process the facial features of individuals without their express permission or consent. Additionally, we do not allow the usage of features of political figures, public officials, or any other public figures without their permission.

We also do not assume any responsibility or liability for the consequences that may arise from the use of our service. Our service is provided on an "as is" basis, and we do not guarantee the accuracy, completeness, or reliability of the results obtained through the use of our service.

By using our service, you agree to indemnify and hold us harmless from any claim or demand, including reasonable attorneys' fees, made by any third-party due to or arising out of your use of the service, your violation of these terms and conditions, or your violation of any rights of another.

In summary, we strictly prohibit the use of our service for any illegal or unethical activities and we are not responsible for any consequences that may arise from the use of our service. If you agree to these terms and conditions, please proceed to use our service.

## License

Our service does not claim any intellectual property rights over the original images or the transformed AI-generated images. Any use of these AI-generated images should respect the copyrights and trademarks of the original images and should not infringe upon the rights of the original copyright owners.

As long as the images do not infringe on any copyrights, paid users can use the generated images for commercial purposes. Free members can not. It is crucial to indicate that these images were altered and generated by Picsi.Ai - Powered by InsightFace, in a visible and accessible manner, to ensure compliance with our licensing terms, legal obligations, and ethical considerations. If a digital picture, this must also be included in the meta and exif data of the photo.

## Introduction

For over 99% of people, using Midjourney to create your own portraits is not feasible unless you're a famous celebrity with thousands or millions of photos online. But now, with the InsightFaceSwap Discord bot, you can accomplish this task easily with just a few steps.

<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/swapd0.jpg" width="800"/>
</div>

## Discord Slash Commands

InsightFaceSwap bot can help you with the following commands:

### /saveid ``name`` ``upload-ID-image``

Used to upload and register your own ID photo or numpy feature for subsequent facial replacement and editing. You can upload up to 10 instances permanently and use them without having to upload them repeatedly.

(Front-view, high quality, no glasses, no heavy bangs ID photos are prefered.）

### /setid ``name/prefer``

This command can be used to do two things.

1) Set default identity name(s), for image generation using context menu. If you need to set multiple ID names, please use commas to separate them.
2) Set prefer options, e.g. use ``/setid --greedy`` to enable greedy mode and ``/setid --nogreedy`` to disable. (The prefer options are placed in the ``idname`` field of ``/setid`` command, don't worry about it)

Note that you can not set current id names and prefer options in one ``/setid`` command simultaneously, but call them separately.

### /listid

List all registered identity names, default identity names and prefer options.

### /delid ``name``

Delete specific identity name.

### /delall 

Delete all registered names.

### /swapid ``name(s)`` ``upload-image``

Replace the face with the registered identity name(s) on target image.

## Discord Context Menu

### Apps/INSwapper

Replace the face with the current/default identity name(s) on target image. Current/default identity name(s) can be set via ``/savevid`` and ``/setid`` slash commands.


   

## Step-by-step guide:

1. Refer to [this link](https://docs.midjourney.com/docs/invite-the-bot) to register Discord app, create a new chat room, and invite the Midjourney bot to the chat room.
2. Invite the InsightFaceSwap bot to the chat room by this link: <https://discord.com/api/oauth2/authorize?client_id=1090660574196674713&permissions=274877945856&scope=bot>.
<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/swapd1.jpg" width="480"/>
</div>
3. Use ``/saveid`` command to register your identity name and feature. Here 'mnls' is the registered name, which can be any alphabets or numbers up to 8 characters long. If everything goes well, the bot will tell you that the save was successful. Note that the newly created identity will be automatically set as the default identity.
<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/swapd2.jpg" width="640"/>
</div>
4. Next, we can experiment with creating the portrait. Let's start chanting the Midjourney prompt and enlarge one of the outputs.
<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/swapd3.jpg" width="640"/>
</div>
5. After the enlargement is complete, we can simply use the ``INSwapper`` context menu to generate our portrait. Right click on the target image and then select ``Apps-INSwapper`` menu. Note that we can also use ``/setid`` command to change the default identity name.
<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/swapd4.jpg" width="640"/>
</div>
6. Generally, the task is completed in less than a second and we can see the result.
<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/swapd5.jpg" width="640"/>
</div>
7. In addition to processing photos generated by Midjourney, we can also process locally uploaded photos by using ``/swapid`` command explicitly.
<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/swapd6.jpg" width="640"/>
</div>
8. Hit to complete!
<div align="left">
<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/swapd7.jpg" width="640"/>
</div>
9. Note that the ``INSwapper`` context menu can also work on user uploaded images in your Discord channel.


## FAQ

Q: Why "application did not respond"?

A: This error indicates that the server was overloaded at the time. Please try again.

Q: Why is the service sometimes slow to respond?

A: We used a queue in our backend. When there are too many users online, the requests will be queued and processed one by one.

Q: Can I list my registered ID list?

A: Yes, use ``/listid`` command.

Q: Are there any restrictions on ID names?

A: All ID names can only be alphabets and numbers, and cannot exceed 10 characters. The total number of registered IDs cannot exceed 20.

Q: Can I delete my registered IDs?

A: You can use ``/delid`` and ``/delall`` commands to delete registered IDs.

Q: Support multi-facial replacement?

A: Yes, you can input a comma splitted idname list, such as ``/setid me,you,him,her``. You can also use the ``_`` symbol to indicate no-replacement(e.g. ``/setid me,_,him``).

Q: How to get good results?

A: 1) Select front-view, high quality, no glasses, no heavy bangs ID photos; 2) Try greedy mode if you need higher identity similarity; 3) For the target image, please ensure that the facial features are proportionate to those of real humans, otherwise it may cause overflow effects.

## Other notes:

1. Front-view, high quality, no glasses, no heavy bangs ID photos are prefered.
2. Each Discord account can execute 50 commands per day.
3. This is in early development stage, so we cannot guarantee that the result will be great in every cases.
4. If there's any problem, please join our Discord group: [link](https://discord.gg/Ym3X8U59ZN)


