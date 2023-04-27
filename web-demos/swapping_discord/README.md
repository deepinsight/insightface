# Using Midjourney and InsightFaceSwap Bot to create a personalized portrait

## Updates

**`2023-04-27`**: Now we support swapping on GIF. The usage is the same as static images. A few extra key points: 1) Uploaded gifs cannot exceed 5MB in size; 2) Performing one gif face swap will consume 5 command opportunities (i.e. a maximum of 10 gifs can be operated per day); 3) Up to the first 15 frames can be operated; 4) The frames may be dynamically resized to a lower resolution.

**`2023-04-18`**: Now we support Discord application commands(AKA. slash commands), please remember joining our [Discord group](https://discord.gg/65Ma47ymPc) to get notification.

## Disclaimer

By using this service, you acknowledge that you have read, understood, and agreed to the terms and conditions outlined in this disclaimer.

We would like to emphasize that our service is intended for research and/or entertainment purposes only. We do not condone or promote the use of our service for any illegal or unethical activities. We strictly prohibit the use of our service to replace faces of individuals without their express permission or consent. Additionally, we do not allow the replacement of faces of political figures, public officials, or any other public figures without their permission.

We also do not assume any responsibility or liability for the consequences that may arise from the use of our service. Our service is provided on an "as is" basis, and we do not guarantee the accuracy, completeness, or reliability of the results obtained through the use of our service.

By using our service, you agree to indemnify and hold us harmless from any claim or demand, including reasonable attorneys' fees, made by any third-party due to or arising out of your use of the service, your violation of these terms and conditions, or your violation of any rights of another.

In summary, our service is provided for research and entertainment purposes only, and we strictly prohibit the use of our service for any illegal or unethical activities. We are not responsible for any consequences that may arise from the use of our service. If you agree to these terms and conditions, please proceed to use our service.

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

### /setid ``name(s)``

Set current/default identity name(s), for image generation using context menu. If you need to set multiple ID names, please use commas to separate them.

### /listid

List all registered identity names.

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



## Other notes:

1. You can use ``/listid`` command to list all the registered IDs. The total number of registered IDs cannot exceed 10. And also you can use ``/delid`` and ``/delall`` commands to delete registered IDs.
2. The registered ID name can only be alphabets and numbers, and cannot exceed 8 characters.
3. For multi-facial replacement, you can input a comma splitted idname list, such as ``/setid me,you,him,her``
4. You can overwrite old ID features by re-uploading with the same ID name.
5. Front-view, high quality, no glasses, no heavy bangs ID photos are prefered.
6. If you don't want to upload your ID photo, you can use the insightface python package to generate your own facial ID features and save them as a .npy file, where shape=(512,), for uploading.
7. Each Discord account can execute 50 commands per day to avoid automated scripts.
8. This is in early development stage, so we cannot guarantee that the result will be great in every cases.
9. Please use it for personal entertainment purposes only.
10. If there's any problem, please join our Discord group: [link](https://discord.gg/65Ma47ymPc)


