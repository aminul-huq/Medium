import torch,os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os
from os import path
import copy

best_acc = 0
best_loss = 10000000

def train(model,train_loader,criterion,optim,device,epochs):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images,labels) in enumerate(tqdm(train_loader)):
                        
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(images)
        outputs = outputs.squeeze()
        
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        train_loss += loss.item() * images.size(0)
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epochs+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))

    return train_loss/len(train_loader), total_correct*100/total

def test(model,test_loader, criterion,optim,modelname,device,epochs):
    model.eval()
    global best_acc
    test_loss,total_correct, total = 0,0,0
    
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = total_correct*100/total
        print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))  

       
        if acc > best_acc:
            print('Saving Best model...')
            state = {
                        'model':model.state_dict(),
                        'acc':acc,
                        'epoch':epochs,
                }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point+modelname+'model.pth.tar')
            best_acc = acc
        
    return test_loss/len(test_loader),acc


def best_test(model,test_loader,criterion,optim,device,epochs):
    model.eval()
    test_loss,total_correct, total = 0,0,0
    y,y_pred,img = [],[],[]
    for i,(images,labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            
    acc = total_correct*100/total
    print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))    
    return test_loss/len(test_loader),acc,y,y_pred



def train_reg(model,train_loader,criterion,optim,train_scheduler,warmup_scheduler,device,epochs):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images,labels) in enumerate(tqdm(train_loader)):
        if epochs > 5:
            train_scheduler.step(epochs)
        if epochs <= 5:
                warmup_scheduler.step()
                
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        train_loss += loss.item() * images.size(0)

    print("Epoch: [{}]  loss: [{:.6f}] ".format(epochs+1,train_loss/len(train_loader)))

    return train_loss/len(train_loader)

def test_reg(model,test_loader, criterion,optim,filename,modelname,device,epochs):
    model.eval()
    global best_loss
    test_loss = 0
    
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            

       
        print("Epoch: [{}]  loss: [{:.6f}]".format(epochs+1,test_loss/len(test_loader)))  

        if filename!=None:
            f = open(filename+".txt","a+")
            f.write("Epoch: [{}]  loss: [{:.6f}]".format(epochs+1,test_loss/len(test_loader)))
            f.close()


        if test_loss/len(test_loader) < best_loss:
            print('Saving Best model...')
            state = {
                        'model':model.state_dict(),
                        'loss':test_loss/len(test_loader),
                        'epoch':epochs,
                }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point+modelname+'model.pth.tar')
            best_loss = test_loss/len(test_loader)

        
        
    return test_loss/len(test_loader)





def trainABN(model,train_loader,criterion,optim,device,epochs,b):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        att_outputs, outputs, _  = model(inputs)
        att_loss = criterion(att_outputs, targets)
        per_loss = criterion(outputs, targets)
        
        if b == True:
            w1, w2 = weighted(att_loss, per_loss)
        else:
            w1 = w2 = 1
        loss = w1*att_loss + w2*per_loss
        
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item() * inputs.size(0)
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epochs+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))

    return train_loss/len(train_loader), total_correct*100/total





def testABN(model,test_loader, criterion,optim,filename,modelname,device,epochs):
    model.eval()
    test_loss, total_correct, total = 0,0,0
    softmax = nn.Softmax(dim=1)
    count = 0
    global best_acc
    
    for i,(inputs, targets) in enumerate(tqdm(test_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            _, outputs, attention = model(inputs)
            outputs = softmax(outputs)
            loss = criterion(outputs, targets)
            attention, fe, per = attention
        
        c_att = attention.data.cpu()
        c_att = c_att.numpy()
        d_inputs = inputs.data.cpu()
        d_inputs = d_inputs.numpy()
        
        in_b, in_c, in_y, in_x = inputs.shape
        for item_img, item_att in zip(d_inputs, c_att):

            v_img = ((item_img.transpose((1,2,0))))* 256
            v_img = v_img[:, :, ::-1]
            resize_att = cv2.resize(item_att[0], (in_x, in_y))
            resize_att *= 255.

            cv2.imwrite('stock1.png', v_img)
            cv2.imwrite('stock2.png', resize_att)
            v_img = cv2.imread('stock1.png')
            vis_map = cv2.imread('stock2.png', 0)
            jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
            jet_map = cv2.add(v_img, jet_map)

            out_dir = path.join('output')
            if not path.exists(out_dir):
                os.mkdir(out_dir)
            #out_path = path.join(out_dir, 'attention', '{0:06d}.png'.format(count))
            fname = '/home/aminul/unr/ABN_corn_21_12_22/val_output/attention_' + str(count) + '.png'
            cv2.imwrite(fname, jet_map)
            #out_path = path.join(out_dir, 'raw', '{0:06d}.png'.format(count))
            fname = '/home/aminul/unr/ABN_corn_21_12_22/val_output/raw_' + str(count)+ '.png'
            cv2.imwrite(fname, v_img)

            count += 1

        test_loss += loss.item() * inputs.size(0)
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
        
    acc = total_correct*100/total
    print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),
                                                                               acc))
    
    if filename!=None:
            f = open(filename+".txt","a+")
            f.write('Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}]\n'.format(epochs+1,test_loss/len(test_loader),acc))
            f.close()


            if acc > best_acc:
                    print('Saving Best model...')
                    state = {
                                'model':model.state_dict(),
                                'acc':acc,
                                'epoch':epochs,
                        }

                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    save_point = './checkpoint/'
                    if not os.path.isdir(save_point):
                        os.mkdir(save_point)

                    torch.save(state, save_point+modelname+'model.pth.tar')
                    best_acc = acc

    return test_loss/len(test_loader), acc


def best_testABN(model,test_loader,criterion,optim,device,epochs):
    model.eval()
    test_loss, total_correct, total = 0,0,0
    softmax = nn.Softmax(dim=1)
    count = 0
    y,y_pred = [],[]
    
    for i,(inputs,targets) in enumerate(tqdm(test_loader)):
        inputs,targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            _, outputs, attention = model(inputs)
            outputs = softmax(outputs)
            loss = criterion(outputs, targets)
            attention, fe, per = attention
        
        c_att = attention.data.cpu()
        c_att = c_att.numpy()
        d_inputs = inputs.data.cpu()
        d_inputs = d_inputs.numpy()
        
        in_b, in_c, in_y, in_x = inputs.shape
        for item_img, item_att in zip(d_inputs, c_att):

            v_img = ((item_img.transpose((1,2,0))))* 256
            v_img = v_img[:, :, ::-1]
            resize_att = cv2.resize(item_att[0], (in_x, in_y))
            resize_att *= 255.

            cv2.imwrite('stock1.png', v_img)
            cv2.imwrite('stock2.png', resize_att)
            v_img = cv2.imread('stock1.png')
            vis_map = cv2.imread('stock2.png', 0)
            jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
            jet_map = cv2.add(v_img, jet_map)

            out_dir = path.join('output')
            if not path.exists(out_dir):
                os.mkdir(out_dir)
            #out_path = path.join(out_dir, 'attention', '{0:06d}.png'.format(count))
            fname = '/home/aminul/unr/ABN_corn_21_12_22/test_output/attention_' + str(count) + '.png'
            cv2.imwrite(fname, jet_map)
            #out_path = path.join(out_dir, 'raw', '{0:06d}.png'.format(count))
            fname = '/home/aminul/unr/ABN_corn_21_12_22/test_output/raw_' + str(count)+ '.png'
            cv2.imwrite(fname, v_img)

            count += 1

        test_loss += loss.item() * inputs.size(0)
        _,predicted = torch.max(outputs.data,1)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
        
        y.append(targets.cpu().numpy())
        y_pred.append(predicted.cpu().numpy())
    acc = total_correct*100/total
    print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))    
    return test_loss/len(test_loader),acc,y,y_pred





def trainMTL(model,train_loader,criterion1,criterion2,optim,train_scheduler,warmup_scheduler,device,epochs):
    model.train()
    clf_loss, reg_loss, tc1, tc2, total1, total2 = 0,0,0,0,0,0

    for i,(images,labels1,labels2) in enumerate(tqdm(train_loader)):
        if epochs > 5:
            train_scheduler.step(epochs)
        if epochs <= 5:
                warmup_scheduler.step()
                
        images, labels1, labels2 = images.to(device), labels1.to(device), labels2.to(device)
        optim.zero_grad()
        op1, op2 = model(images)
        
        
        loss1 = criterion1(op1, labels1)
        loss2 = criterion2(op2, labels2)
        
        print(loss1, loss2)
    
        # a, b = exp_weighted(loss1,loss2)
        loss = loss1 + loss2        
       
        
        loss.backward()
        optim.step()

        clf_loss += loss1.item() * images.size(0)
        reg_loss += loss2.item() * images.size(0)
        
        _,pd1 = torch.max(op1.data,1)
        # _,pd2 = torch.max(op2.data,1)

        tc1 += (pd1 == labels1).sum().item()
        # tc2 += (pd2 == labels2).sum().item()
        
        total1 += labels1.size(0)
        # total2 += labels2.size(0)

    print("Epoch: [{}]  Classification loss: [{:.2f}] Classification Accuracy [{:.2f}] Regression loss: [{:.6f}]".format(epochs+1,clf_loss/len(train_loader),
                                                                               tc1*100/total1,reg_loss/len(train_loader) ))

    return clf_loss/len(train_loader), tc1*100/total1, reg_loss/len(train_loader) 

def testMTL(model,test_loader, criterion1,criterion2,optim,filename,modelname,device,epochs):
    model.eval()
    global best_loss
    clf_loss, reg_loss, tc1, tc2, total1, total2 = 0,0,0,0,0,0
    
    with torch.no_grad():
        for i,(images,labels1,labels2) in enumerate(tqdm(test_loader)):
            images, labels1, labels2 = images.to(device), labels1.to(device), labels2.to(device)
            op1, op2 = model(images)
            loss1 = criterion1(op1, labels1)
            loss2 = criterion2(op2, labels2)
            loss = loss1 + loss2

            clf_loss += loss1.item() * images.size(0)
            reg_loss += loss2.item() * images.size(0)
            
            _,pd1 = torch.max(op1.data,1)
            # _,pd2 = torch.max(op2.data,1)

            tc1 += (pd1 == labels1).sum().item()
            # tc2 += (pd2 == labels2).sum().item()

            total1 += labels1.size(0)
            # total2 += labels2.size(0)

        acc1 = tc1*100/total1
        # acc2 = tc2*100/total2
        print("Epoch: [{}]  Classification loss: [{:.2f}] Classification Accuracy [{:.2f}] Regression loss: [{:.6f}]".format(epochs+1,clf_loss/len(test_loader),
                                                                               tc1*100/total1,reg_loss/len(test_loader) ))
        if filename!=None:
            f = open(filename+".txt","a+")
            f.write("Epoch: [{}]  Classification loss: [{:.2f}] Classification Accuracy [{:.2f}] Regression loss: [{:.6f}]".format(epochs+1,clf_loss/len(test_loader),
                                                                               tc1*100/total1,reg_loss/len(test_loader) ))
            f.close()


        if loss1+loss2 < best_loss:
            print('Saving Best model...')
            state = {
                        'model':model.state_dict(),
                        # 'acc':acc,
                        'epoch':epochs,
                }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point+modelname+'model.pth.tar')
            best_loss = loss1+loss2

        
        
    return clf_loss/len(test_loader), tc1*100/total1,reg_loss/len(test_loader) 


def best_testMTL(model,test_loader,criterion1,criterion2,optim,device,epochs):
    model.eval()
    clf_loss, reg_loss, tc1, tc2, total1, total2 = 0,0,0,0,0,0
    y_true1,y_true2,y_pred1,y_pred2 = [],[],[],[]
    for i,(images,labels1,labels2) in enumerate(tqdm(test_loader)):
        images, labels1, labels2 = images.to(device), labels1.to(device), labels2.to(device)
        op1, op2 = model(images)
        loss1 = criterion1(op1, labels1)
        loss2 = criterion2(op1, labels2)
        
        
        # a,b = weighted(loss1, loss2)
        
        loss = a*loss1 + b*loss2
        
        clf_loss += loss1.item() * images.size(0)
        reg_loss += loss2.item() * images.size(0)
        
        _,pd1 = torch.max(op1.data,1)
        # _,pd2 = torch.max(op2.data,1)

        tc1 += (pd1 == labels1).sum().item()
        # tc2 += (pd2 == labels2).sum().item()

        total1 += labels1.size(0)
        # total2 += labels2.size(0)
        
        y_true1.append(labels1.cpu().numpy())
        # y_true2.append(labels2.cpu().numpy())
        y_pred1.append(pd1.cpu().numpy())
        # y_pred2.append(pd2.cpu().numpy())
        
    acc1 = tc1*100/total1
    # acc2 = tc2*100/total2
    print("Epoch: [{}]  Classification loss: [{:.2f}] Classification Accuracy [{:.2f}] Regression loss: [{:.6f}]".format(epochs+1,clf_loss/len(test_loader),
                                                                               tc1*100/total1,reg_loss/len(test_loader) ))
    return acc1,y_true1,y_pred1





def weighted(a_loss, t_loss):
    a,b = copy.deepcopy(a_loss.detach()), copy.deepcopy(t_loss.detach())
    
    total = a + b
    w1 = a / total
    w2 = b / total
    
    return w1*2, w2*2
    
def exp_weighted(a_loss, t_loss):
    a,b = copy.deepcopy(a_loss.detach().cpu().numpy()), copy.deepcopy(t_loss.detach().cpu().numpy())
    
    total = np.exp(a) + np.exp(b)
    w1 = np.exp(a) / total
    w2 = np.exp(b) / total
    
    return w1*2, w2*2
    
    