#import "CoreMLAdapter.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

@interface CoreMLAdapterImpl : NSObject
@property (nonatomic, strong) MLModel *model;
@property (nonatomic, strong) id<MLFeatureProvider> inputFeatures;
@property (nonatomic, strong) id<MLFeatureProvider> outputFeatures;
@property (nonatomic, strong) NSURL *modelURL;
@end

@implementation CoreMLAdapterImpl
@end // CoreMLAdapterImpl


class CoreMLAdapter::Impl {
public:
    CoreMLAdapterImpl *impl;
};


CoreMLAdapter::CoreMLAdapter() : pImpl(new Impl) {
    pImpl->impl = [[CoreMLAdapterImpl alloc] init];
}

CoreMLAdapter::~CoreMLAdapter() {
    delete pImpl;
}

int32_t CoreMLAdapter::readFromFile(const std::string &modelPath) {
    NSString *modelPathStr = [NSString stringWithUTF8String:modelPath.c_str()];
    NSURL *modelURL = [NSURL fileURLWithPath:modelPathStr];    
    NSError *error = nil;
    pImpl->impl.model = [MLModel modelWithContentsOfURL:modelURL error:&error];
    pImpl->impl.modelURL = modelURL;
    if (error) {
        NSLog(@"Error loading model: %@", error);
        return COREML_HFAIL;
    }
    // printModelInfo();
    return COREML_HSUCCEED;
}


CoreMLAdapter CoreMLAdapter::readNetFrom(const std::string& modelPath) {
    CoreMLAdapter net;
    NSString *modelPathStr = [NSString stringWithUTF8String:modelPath.c_str()];
    NSURL *modelURL = [NSURL fileURLWithPath:modelPathStr];
    NSError *error = nil;
    net.pImpl->impl.model = [MLModel modelWithContentsOfURL:modelURL error:&error];
    net.pImpl->impl.modelURL = modelURL;
    if (error) {
        NSLog(@"Error loading model: %@", error);
    }
    return net;
}


CoreMLAdapter CoreMLAdapter::readNetFromBin(const std::vector<char>& model_data) {
    // Currently, reading binary files directly is not supported!
    CoreMLAdapter net;
    NSLog(@"readNetFromBin not fully implemented");
    return net;
}

std::vector<std::string> CoreMLAdapter::getInputNames() const {
    std::vector<std::string> names;
    for (NSString *name in pImpl->impl.model.modelDescription.inputDescriptionsByName.allKeys) {
        names.push_back([name UTF8String]);
    }
    return names;
}


std::vector<std::string> CoreMLAdapter::getOutputNames() const {
    std::vector<std::string> names;
    for (NSString *name in pImpl->impl.model.modelDescription.outputDescriptionsByName.allKeys) {
        names.push_back([name UTF8String]);
    }
    return names;
}

std::vector<int> CoreMLAdapter::getInputShapeByName(const std::string &name) {
    NSString *nsName = [NSString stringWithUTF8String:name.c_str()];
    MLFeatureDescription *desc = pImpl->impl.model.modelDescription.inputDescriptionsByName[nsName];
    
    if (desc.type != MLFeatureTypeMultiArray) {
        return {};
    }
    std::vector<int> shape;
    for (NSNumber *dim in desc.multiArrayConstraint.shape) {
        shape.push_back(dim.intValue);
    }
    return shape;
}

std::vector<int> CoreMLAdapter::getOutputShapeByName(const std::string &name) {
    if (m_outputShapes.find(name) != m_outputShapes.end()) {
        return m_outputShapes[name];
    }

    return {};
}


void CoreMLAdapter::setInput(const char* inputName, const char* data) {
    NSString *nsInputName = [NSString stringWithUTF8String:inputName];
    MLFeatureDescription *desc = pImpl->impl.model.modelDescription.inputDescriptionsByName[nsInputName];
    if (desc.type != MLFeatureTypeMultiArray) {
        NSLog(@"Input %s is not a MultiArray", inputName);
        return;
    }
    
    NSArray<NSNumber *> *shape = desc.multiArrayConstraint.shape;
    NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:shape.count];
    int stride = 1;
    for (NSInteger i = shape.count - 1; i >= 0; i--) {
        [strides insertObject:@(stride) atIndex:0];
        stride *= [shape[i] intValue];
    }
    
    NSError *error = nil;
    MLMultiArray *multiArray = [[MLMultiArray alloc] initWithDataPointer:(void*)data
                                                                   shape:shape
                                                                dataType:MLMultiArrayDataTypeFloat32
                                                                 strides:strides
                                                             deallocator:nil
                                                                   error:&error];
    if (error) {
        NSLog(@"Error creating MLMultiArray: %@", error);
        return;
    }
    
    MLFeatureValue *featureValue = [MLFeatureValue featureValueWithMultiArray:multiArray];
    pImpl->impl.inputFeatures = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{nsInputName: featureValue} error:&error];
    if (error) {
        NSLog(@"Error creating MLDictionaryFeatureProvider: %@", error);
    }
}


int32_t CoreMLAdapter::forward() {
    @autoreleasepool {
        if (!pImpl->impl.inputFeatures) {
            NSLog(@"Input features not set");
            return COREML_FORWARD_FAILED;
        }
        NSError *error = nil;
        pImpl->impl.outputFeatures = [pImpl->impl.model predictionFromFeatures:pImpl->impl.inputFeatures
                                                                    options:[[MLPredictionOptions alloc] init]
                                                                        error:&error];
        if (error) {
            NSLog(@"Error in forward pass: %@", error);
            return COREML_FORWARD_FAILED;
        }

        m_outputShapes.clear();
        for (NSString *outputName in [pImpl->impl.outputFeatures featureNames]) {
            MLFeatureValue *value = [pImpl->impl.outputFeatures featureValueForName:outputName];
            if (value.multiArrayValue) {
                NSArray<NSNumber *> *shapeArray = value.multiArrayValue.shape;
                std::vector<int> shapeVector;
                
                for (NSNumber *dim in shapeArray) {
                    shapeVector.push_back([dim intValue]);
                }
                
                std::string outputNameStr = [outputName UTF8String];
                m_outputShapes[outputNameStr] = shapeVector;
            }
        }
        return COREML_HSUCCEED;
    }
}


const char* CoreMLAdapter::getOutput(const char* nodeName) {
    @autoreleasepool {
        if (!pImpl->impl.outputFeatures) {
            NSLog(@"No output features available. Did you call forward()?");
            return nullptr;
        }
        NSString *nsNodeName = [NSString stringWithUTF8String:nodeName];
        MLFeatureValue *featureValue = [pImpl->impl.outputFeatures featureValueForName:nsNodeName];
        if (featureValue.type != MLFeatureTypeMultiArray) {
            NSLog(@"Error getting output for node '%s': Output is not a MultiArray. Current type: %@", nodeName, @(featureValue.type));
            return nullptr;
        }
        return (const char*)featureValue.multiArrayValue.dataPointer;
    }
}


void CoreMLAdapter::setInferenceMode(InferenceMode mode) {
    MLComputeUnits computeUnits;
    switch (mode) {
        case InferenceMode::CPU:
            computeUnits = MLComputeUnitsCPUOnly;
            break;
        case InferenceMode::GPU:
            computeUnits = MLComputeUnitsCPUAndGPU;
            break;
        case InferenceMode::ANE:
            computeUnits = MLComputeUnitsAll;
            break;
    }
    
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = computeUnits;
    
    NSError *error = nil;
    pImpl->impl.model = [MLModel modelWithContentsOfURL:pImpl->impl.modelURL
                                          configuration:config
                                                  error:&error];
    if (error) {
        NSLog(@"Error setting inference mode: %@", error);
    }
}


void CoreMLAdapter::printModelInfo() const {
    NSLog(@"Model Input Description:");
    for (NSString *inputName in pImpl->impl.model.modelDescription.inputDescriptionsByName) {
        MLFeatureDescription *desc = pImpl->impl.model.modelDescription.inputDescriptionsByName[inputName];
        NSLog(@"Input Name: %@", inputName);
        NSLog(@"Input Type: %@", @(desc.type));
        
        switch (desc.type) {
            case MLFeatureTypeMultiArray:
                NSLog(@"MLFeatureTypeMultiArray");
                NSLog(@"  Shape: %@", desc.multiArrayConstraint.shape);
                NSLog(@"  Data Type: %@", @(desc.multiArrayConstraint.dataType));
                break;
            case MLFeatureTypeImage:
                NSLog(@"MLFeatureTypeImage");
                NSLog(@"  Image Size: %dx%d", (int)desc.imageConstraint.pixelsWide, (int)desc.imageConstraint.pixelsHigh);
                NSLog(@"  Color Space: %@", desc.imageConstraint.pixelFormatType == kCVPixelFormatType_32BGRA ? @"BGRA" : @"Other");
                break;
            case MLFeatureTypeDictionary:
                NSLog(@"  Dictionary Key Type: %@", @(desc.dictionaryConstraint.keyType));
                break;
            case MLFeatureTypeSequence:
                NSLog(@"  Sequence Constraint: %@", desc.sequenceConstraint);
                break;
            default:
                NSLog(@"  Unknown type details");
                break;
        }
    }
    
    NSLog(@"Model Output Description:");
    for (NSString *outputName in pImpl->impl.model.modelDescription.outputDescriptionsByName) {
        MLFeatureDescription *desc = pImpl->impl.model.modelDescription.outputDescriptionsByName[outputName];
        NSLog(@"Output Name: %@", outputName);
        NSLog(@"Output Type: %@", @(desc.type));
        
        switch (desc.type) {
            case MLFeatureTypeMultiArray:
                NSLog(@"  Shape: %@", desc.multiArrayConstraint.shape);
                NSLog(@"  Data Type: %@", @(desc.multiArrayConstraint.dataType));
                break;
            case MLFeatureTypeImage:
                NSLog(@"  Image Size: %dx%d", (int)desc.imageConstraint.pixelsWide, (int)desc.imageConstraint.pixelsHigh);
                NSLog(@"  Color Space: %@", desc.imageConstraint.pixelFormatType == kCVPixelFormatType_32BGRA ? @"BGRA" : @"Other");
                break;
            case MLFeatureTypeDictionary:
                NSLog(@"  Dictionary Key Type: %@", @(desc.dictionaryConstraint.keyType));
                break;
            case MLFeatureTypeSequence:
                NSLog(@"  Sequence Constraint: %@", desc.sequenceConstraint);
                break;
            default:
                NSLog(@"  Unknown type details");
                break;
        }
    }
}
